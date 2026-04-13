import json
import logging
import gc
import types
import sys
import os
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union
import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils_hf import get_dtype
from lm_eval.__main__ import cli_evaluate


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from transformers import AutoTokenizer
from modeling.llada import LLaDAModelLM

from generation.generation_core import DLMGeneration
from generation.determinism_utils import setup_deterministic_env
from generation.monitor_utils import ForwardMonitor


eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")

# Last constructed LLaDA LM (per process). Used to merge generation stats into lm_eval JSON.
_ACTIVE_LLADA_LM: Optional["Llada"] = None


def _inference_summary_table(summary: dict) -> str:
    """One header row (metric names) + one value row; middle row is Markdown ``---`` separator."""
    keys: List[str] = []
    vals: List[str] = []
    for k, v in summary.items():
        if v is None:
            continue
        keys.append(str(k).replace("|", "/"))
        val = f"{v:.6g}" if isinstance(v, float) else str(v)
        vals.append(val.replace("|", "/"))
    if not keys:
        return ""
    cell_widths = [max(len(key), len(val)) for key, val in zip(keys, vals)]
    return (
        "\n|"
        + "|".join(key.center(width) for key, width in zip(keys, cell_widths))
        + "|\n|"
        + "|".join("-" * width for width in cell_widths)
        + "|\n|"
        + "|".join(val.rjust(width) for val, width in zip(vals, cell_widths))
        + "|\n"
    )


@register_model("llada")
class Llada(LM):
    _lm_eval_save_hook_installed: bool = False
    _make_table_hook_installed: bool = False

    @staticmethod
    def _emit_inference_stats_for_eval(
        inst: Optional["Llada"],
        lm_eval_results: Optional[dict],
    ) -> None:
        """Merge LLaDA generation stats into lm_eval output: dict key, log line, optional sidecar JSON.

        Called once when lm_eval flushes aggregated results (see ``_install_lm_eval_save_hook``).
        """
        if inst is None or getattr(inst, "_rank", 0) != 0:
            return
        ts = inst.inference_stats["inference_time"]
        if not ts:
            return

        nf = inst.inference_stats["nfe"]
        tok = inst.inference_stats["generated_tokens"]

        def _scalar_int(x):
            if hasattr(x, "item"):
                return int(x.item())
            return int(x)

        def _scalar_float(x):
            if hasattr(x, "item"):
                return float(x.item())
            return float(x)

        tok_i = [_scalar_int(t) for t in tok]
        nf_i = [_scalar_int(x) for x in nf]
        ts_f = [_scalar_float(x) for x in ts]

        # n = len(ts_f)
        total_time = sum(ts_f)
        total_nfe = sum(nf_i)
        total_tok = sum(tok_i)
        tpf = (total_tok / total_nfe) if total_nfe else None
        payload: dict = {
            "summary": {
                # "num_generation_batches": n,
                "Total Time (s)": total_time,
                "Total NFE": total_nfe,
                "Total Generated Tokens": total_tok,
                "Avg Fwd time (ms)": total_time * 1000 / total_nfe,
                "Avg TPS": total_tok / total_time,
                "Avg TPF": tpf,
            },
            "per_batch": {
                "inference_time_s": ts_f,
                "nfe": nf_i,
                "generated_tokens": tok_i,
            },
        }
        if torch.cuda.is_available():
            payload["summary"]["GPU Peak Mem (MB)"] = (
                torch.cuda.max_memory_reserved()
                / (1024**2)
            )
        if getattr(inst, "_world_size", 1) > 1:
            payload = {
                **payload,
                "note": (
                    "Multi-GPU data parallel: stats are only from this process's "
                    "generation work (often an incomplete shard). Use one GPU for "
                    "full-accounting totals or merge per-rank stats yourself."
                ),
            }

        if (
            lm_eval_results is not None
            and getattr(inst, "merge_inference_stats_into_lm_eval_results", True)
        ):
            lm_eval_results["llada_inference_stats"] = payload

        if getattr(inst, "_print_inference_stats_at_exit", True):
            eval_logger.info(
                "LLaDA inference stats (summary): %s",
                json.dumps(payload["summary"]),
            )

        path = getattr(inst, "_inference_stats_json", None)
        if path:
            if getattr(inst, "_world_size", 1) > 1:
                root, ext = os.path.splitext(path)
                path = f"{root}.rank{getattr(inst, '_rank', 0)}{ext}"
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            eval_logger.info("Wrote LLaDA inference stats to %s", path)

    @classmethod
    def _install_lm_eval_save_hook(cls) -> None:
        """Wrap ``EvaluationTracker.save_results_aggregated`` once (no LM callback in lm_eval).

        The wrapper calls ``_emit_inference_stats_for_eval`` so metrics JSON, logging, and
        optional sidecar file stay in one code path.
        """
        if cls._lm_eval_save_hook_installed:
            return
        try:
            from lm_eval.loggers.evaluation_tracker import EvaluationTracker
        except ImportError:
            eval_logger.debug(
                "lm_eval.evaluation_tracker missing; cannot hook aggregated results save"
            )
            return

        _orig_save = EvaluationTracker.save_results_aggregated

        def save_results_aggregated(
            tracker_self,
            results: dict,
            samples: Optional[dict] = None,
        ):
            Llada._emit_inference_stats_for_eval(_ACTIVE_LLADA_LM, results)
            return _orig_save(tracker_self, results, samples=samples)

        EvaluationTracker.save_results_aggregated = save_results_aggregated
        cls._lm_eval_save_hook_installed = True

    @classmethod
    def _install_make_table_hook(cls) -> None:
        """Append ``llada_inference_stats`` to the CLI table printed by ``make_table``.

        ``Run._execute`` binds ``make_table`` before ``simple_evaluate``, so this must run
        when this module is loaded (see module-level calls below), not only in ``__init__``.
        """
        if cls._make_table_hook_installed:
            return
        try:
            import lm_eval.utils as lm_eval_utils
        except ImportError:
            eval_logger.debug("lm_eval.utils missing; cannot hook make_table")
            return

        _orig_make_table = lm_eval_utils.make_table

        def make_table(
            result_dict,
            column: str = "results",
            sort_results: bool = False,
        ):
            table = _orig_make_table(
                result_dict, column=column, sort_results=sort_results
            )
            if column != "results":
                return table
            stats = result_dict.get("llada_inference_stats")
            if not stats or not isinstance(stats, dict):
                return table
            summary = stats.get("summary")
            if not summary:
                return table
            extra = _inference_summary_table(summary)
            if not extra:
                return table
            return table.rstrip() + extra

        lm_eval_utils.make_table = make_table  # type: ignore[method-assign]
        cls._make_table_hook_installed = True

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_new_tokens: Optional[int] = 512,
        max_length: Optional[int] = 4096,
        add_bos_token: Optional[bool] = False,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        mc_num: Optional[int] = 128,
        classifier_free_guidance: Optional[float] = 1.0,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 128,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        alg: Optional[str] = "entropy",
        alg_temp: Optional[float] = 0.0,
        escape_until: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        self.max_length = max_length
        self.add_bos_token = add_bos_token
        # generation params
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.alg = alg
        self.alg_temp = alg_temp
        self.escape_until = escape_until

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.mc_num = mc_num
        self.classifier_free_guidance = classifier_free_guidance
        self.sampling_eps = sampling_eps

        self.block_length = kwargs.get('block_length', 32)
        self.dual_cache = kwargs.get('dual_cache', True)
        self.decoding_alg = kwargs.get('decoding_alg', 'base')
        self.early_exit = kwargs.get('early_exit', False)

        self.confidence_threshold = kwargs.get('confidence_threshold', None)

        if self.decoding_alg == 'freedave':
            self.draft_steps = kwargs.get('draft_steps', 4)
            self.draft_mode = kwargs.get('draft_mode', 'tree_attention')
            self.eager_acceptance_mode = kwargs.get('eager_acceptance_mode', False)
        else:
            self.draft_steps = None
            self.draft_mode = None
            self.eager_acceptance_mode = None

        self.deterministic = kwargs.get('deterministic', False)
        self.sdpa_backend = kwargs.get('sdpa_backend', None)
        if self.deterministic:
            setup_deterministic_env(seed=kwargs.get('seed', 42))

        self.dlm_generation = DLMGeneration(
            sdpa_additive_attention_mask=True,
            deterministic=self.deterministic,
            sdpa_backend=self.sdpa_backend,
        )
        self.forward_monitor = ForwardMonitor(self.model)
        
        self.inference_stats = {
            "inference_time": [],
            "nfe": [],
            "generated_tokens": []
        }

        # Optional sidecar JSON / log line: handled in ``_emit_inference_stats_for_eval`` when lm_eval saves.
        # Example: --model_args inference_stats_json=evals_results/run_stats.json
        # Or: export LLADA_INFERENCE_STATS_JSON=...
        self._inference_stats_json = kwargs.get("inference_stats_json") or os.environ.get(
            "LLADA_INFERENCE_STATS_JSON"
        )
        self._print_inference_stats_at_exit = kwargs.get("print_inference_stats_at_exit", True)
        self.merge_inference_stats_into_lm_eval_results = kwargs.get(
            "merge_inference_stats_into_lm_eval_results", True
        )
        global _ACTIVE_LLADA_LM
        _ACTIVE_LLADA_LM = self

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        self.model = (
            LLaDAModelLM.from_pretrained(
                pretrained,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
            )
            .eval()
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        if self.add_bos_token:
            prompts = [self.tokenizer.bos_token + p for p in prompts]
        # tokenize
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        if len(prompt_ids) > self.max_length-self.max_new_tokens:
            eval_logger.warning(f"Prompt length {len(prompt_ids)} is larger than {self.max_length-self.max_new_tokens}, cutoff on the left side")
            prompt_ids = prompt_ids[-(self.max_length-self.max_new_tokens):]

        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        if self.decoding_alg == 'base':
            with self.forward_monitor.count():
                outputs = self.dlm_generation.block_decode_with_full_attention(
                    model=self.model,
                    input_ids=prompt_ids,
                    attention_mask=attn_mask,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    alg_temp=self.alg_temp,
                    block_length=self.block_length,
                    max_gen_length=self.max_new_tokens,
                    decoding_steps=self.diffusion_steps,
                    use_cache=True,
                    dual_cache=self.dual_cache,
                    mask_token_id=self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>"),
                    eos_token_id=self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    pad_token_id=self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    early_exit=self.early_exit,
                )
        elif self.decoding_alg == 'freedave':
            with self.forward_monitor.count():
                outputs = self.dlm_generation.block_decode_with_full_attention_FreeDave(
                    model=self.model,
                    input_ids=prompt_ids,
                    attention_mask=attn_mask,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    alg_temp=self.alg_temp,
                    block_length=self.block_length,
                    max_gen_length=self.max_new_tokens,
                    decoding_steps=self.diffusion_steps,
                    use_cache=True,
                    dual_cache=self.dual_cache,
                    eager_acceptance_mode=self.eager_acceptance_mode,
                    draft_steps=self.draft_steps,
                    draft_mode=self.draft_mode,
                    mask_token_id=self.tokenizer.convert_tokens_to_ids("<|mdm_mask|>"),
                    eos_token_id=self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    pad_token_id=self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                    early_exit=self.early_exit,
                )
        else:
            raise NotImplementedError('Decoding algorithm not supported: {}'.format(self.decoding_alg))

        # decode
        responses = [
            self.tokenizer.decode(g[len(p) :].tolist()).split(self.tokenizer.eos_token)[0]
            for p, g in zip(prompt_ids, outputs.sequences)
        ]

        self.inference_stats["inference_time"].append(self.forward_monitor.get_elapsed_time())
        self.inference_stats["nfe"].append(self.forward_monitor.get_nfe())
        self.inference_stats["generated_tokens"].append(
            int((outputs.trajectory_step_map >= 0).sum().item())
        )
        # self.inference_stats["generated_tokens"].append(sum([g[len(p) :].numel() for p, g in zip(prompt_ids, generation_ids)]))
        self.forward_monitor.reset()
        torch.cuda.empty_cache()

        return responses

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):
        res = []

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            responses = self._generate_batch(contexts)
            if not self.escape_until:
                for i, r in enumerate(responses):
                    for s in gen_args[0]['until']:
                        r = r.split(s)[0]
                    responses[i] = r

            # if self.rank == 0:
            #     print(f"Context:\n{contexts[0]}\nResponse:\n{responses[0]}\n")

            res.extend(responses)
            pbar.update(len(contexts))

        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.tokenizer.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        """
        Full-attention LLaDA: raw ``model(...).logits`` without Dream-style AR logits shift.
        """
        if self.classifier_free_guidance > 1.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.tokenizer.mask_token_id
            batch = torch.cat([batch, un_batch])

        inp = batch

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(inp).logits

        if self.classifier_free_guidance > 1.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.classifier_free_guidance * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        
        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            # eval_logger.info("before noising")
            perturbed_seq_, p_mask = self._forward_process(seq)
            # eval_logger.info("end noising")
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.tokenizer.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context
            
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [self.tokenizer.eos_token_id]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        # by default truncate on the left
        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(f"Text length {len(whole_enc)} is larger than {self.max_length}, cutoff on the left side")
            context_remain = context_enc_len-cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning(f"All context (prompt) is truncated.")
                context_enc = ""
                continuation_enc = whole_enc[-self.max_length:]
        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        print(ds[0])
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                # likelihood calculations are modified from https://github.com/ML-GSAI/SMDM/blob/main/evaluate_diff.py
                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError


Llada._install_make_table_hook()
Llada._install_lm_eval_save_hook()

if __name__ == "__main__":
    cli_evaluate()
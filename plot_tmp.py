import matplotlib.pyplot as plt
import matplotlib as mpl

# Set academic style
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['mathtext.fontset'] = 'dejavuserif'

draft_steps = [1, 2, 4, 8, 16, 32]
accuracy = [74.20, 73.40, 72.20, 76.40, 74.60, 74.20]
throughput_over_time = [7.26, 10.68, 14.68, 16.36, 19.89, 16.84]
throughput_over_nfe = [0.26, 0.37, 0.50, 0.67, 0.78, 0.86]
gpu_mem_peak_usage = [21.65, 22.82, 26.58, 40.56, 97.38, 91.08]

baseline_accuracy = 68.80
baseline_throughput_over_time = 18.94
baseline_throughput_over_nfe = 0.61
baseline_gpu_mem_peak_usage = 21.65

# Professional color palette
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

fig, axs = plt.subplots(1, 4, figsize=(20, 4))

# Helper function to style axes
def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2, length=5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

# Accuracy plot
axs[0].bar(range(len(draft_steps)), accuracy, width=0.65, color=colors[0], edgecolor='black', linewidth=0.8)
axs[0].set_ylabel('Accuracy (%)', fontsize=13, fontweight='normal')
axs[0].set_xlabel('Draft Steps', fontsize=13, fontweight='normal')
axs[0].set_xticks(range(len(draft_steps)))
axs[0].set_xticklabels([str(ds) for ds in draft_steps], fontsize=12)
axs[0].set_ylim(0, 100)
axs[0].tick_params(axis='y', labelsize=12)
style_axis(axs[0])

# Throughput over time
axs[1].bar(range(len(draft_steps)), throughput_over_time, width=0.65, color=colors[1], edgecolor='black', linewidth=0.8)
axs[1].set_ylabel('Throughput over time (#tokens/s)', fontsize=13, fontweight='normal')
axs[1].set_xlabel('Draft Steps', fontsize=13, fontweight='normal')
axs[1].set_xticks(range(len(draft_steps)))
axs[1].set_xticklabels([str(ds) for ds in draft_steps], fontsize=12)
axs[1].set_ylim(0, 22)
axs[1].tick_params(axis='y', labelsize=12)
style_axis(axs[1])

# Throughput over nfe
axs[2].bar(range(len(draft_steps)), throughput_over_nfe, width=0.65, color=colors[2], edgecolor='black', linewidth=0.8)
axs[2].set_ylabel('Throughput over NFEs (#tokens)', fontsize=13, fontweight='normal')
axs[2].set_xlabel('Draft Steps', fontsize=13, fontweight='normal')
axs[2].set_xticks(range(len(draft_steps)))
axs[2].set_xticklabels([str(ds) for ds in draft_steps], fontsize=12)
axs[2].set_ylim(0, 1)
axs[2].tick_params(axis='y', labelsize=12)
style_axis(axs[2])

# GPU memory peak usage
axs[3].bar(range(len(draft_steps)), gpu_mem_peak_usage, width=0.65, color=colors[3], edgecolor='black', linewidth=0.8)
axs[3].set_ylabel('GPU Memory Peak Usage (%)', fontsize=13, fontweight='normal')
axs[3].set_xlabel('Draft Steps', fontsize=13, fontweight='normal')
axs[3].set_xticks(range(len(draft_steps)))
axs[3].set_xticklabels([str(ds) for ds in draft_steps], fontsize=12)
axs[3].set_ylim(0, 100)
axs[3].tick_params(axis='y', labelsize=12)
style_axis(axs[3])

plt.tight_layout()
plt.savefig('plot_tmp.pdf', dpi=300, bbox_inches='tight')
plt.savefig('plot_tmp.png', dpi=300, bbox_inches='tight')
plt.show()
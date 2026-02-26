"""
Generate publication-quality graphs for the combined paper.
Uses REAL benchmark data from RPi4 measurements (Feb 26, 2026).

Run: python generate_paper_graphs.py
Output: ./paper_figures/ directory with PNG files
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ─── Output directory ───────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Color palette ──────────────────────────────────────────────────
C_STT   = "#2196F3"   # blue
C_LLM   = "#FF9800"   # orange
C_TTS   = "#4CAF50"   # green
C_TOTAL = "#9C27B0"   # purple
C_GRAY  = "#757575"
C_RED   = "#F44336"

# Consistent style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.pad_inches': 0.15,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Apply bbox_inches='tight' in each savefig call instead

# ═══════════════════════════════════════════════════════════════════
# REAL BENCHMARK DATA (timeline_20260226_200637.json, N=10 each)
# ═══════════════════════════════════════════════════════════════════

versions = ["V1", "V2", "V3", "V3-opt", "V4"]

# Mean latencies (ms)
stt_mean  = [11821, 11807, 4653, 2566, 4545]
llm_mean  = [0,     0,     5299, 5464, 4337]
tts_mean  = [2535,  2542,  4029, 4726, 4144]
total_mean= [14356, 14349, 13981,12757,13026]

# Std deviations (ms)
stt_std   = [56,    49,    35,   45,   87]
llm_std   = [0,     0,     4579, 3953, 2509]
tts_std   = [27,    34,    1415, 1316, 608]
total_std = [70,    56,    5086, 3998, 2579]

# V4 varied-query benchmark (benchmark_voice.py, 10 different queries)
v4_varied_total_mean = 15818
v4_varied_total_std  = 7940

# V4 per-trial data (varied-query benchmark)
v4_trials_query = [
    "Hello, what is\nthe weather?",
    "Fun fact\nabout robots",
    "What time\nis it?",
    "How does a\ncomputer work?",
    "What is\nyour name?",
    "Tell me a\nshort joke",
    "What is\nartificial\nintelligence?",
    "How far is\nthe moon?",
    "Capital\nof Egypt?",
    "Say something\nnice"
]
v4_trials_stt = [4588, 4619, 4329, 4394, 4357, 4476, 4332, 4411, 4472, 4248]
v4_trials_llm = [15899, 8312, 2158, 8884, 2423, 2705, 8822, 3715, 1888, 2022]
v4_trials_tts = [4734, 10854, 2103, 12622, 2937, 2528, 13706, 3538, 1995, 2110]
v4_trials_total = [25220, 23785, 8590, 25900, 9717, 9710, 26860, 11664, 8355, 8380]

# YOLO detection progression
yolo_phases = ["P1\nPyTorch\nBaseline", "P2\nPyTorch\nOptimized", "P3\nTFLite\nINT8",
               "P4\nPipeline\nOptimized", "P5\nINT8\nFailure", "P6\nYOLOv8\nFP16", "P7\n224px\nFinal"]
yolo_fps = [2, 4, 11, 15, 0, 10, 20]


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Total Latency Comparison Across All 5 Versions
# ═══════════════════════════════════════════════════════════════════

def fig1_total_latency():
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(versions))
    width = 0.6

    # Color V1/V2 differently (no LLM) vs V3+ (with LLM)
    colors = [C_GRAY, C_GRAY, C_TOTAL, C_TOTAL, C_TOTAL]

    bars = ax.bar(x, [t/1000 for t in total_mean], width,
                  yerr=[s/1000 for s in total_std],
                  color=colors, edgecolor='white', linewidth=1.5,
                  capsize=6, error_kw={'linewidth': 2})

    # Add value labels on bars
    for bar, m, s in zip(bars, total_mean, total_std):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{m/1000:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.8,
                f'±{s/1000:.1f}s', ha='center', va='top', fontsize=9, color='white', fontweight='bold')

    # Annotations
    ax.annotate('No LLM\n(keyword only)', xy=(0.5, 14.4), fontsize=10,
                ha='center', va='bottom', color=C_GRAY,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5', edgecolor=C_GRAY))
    ax.annotate('With LLM\n(conversational AI)', xy=(3, 13.5), fontsize=10,
                ha='center', va='bottom', color=C_TOTAL,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e5f5', edgecolor=C_TOTAL))

    # Best version marker
    best_idx = total_mean.index(min(total_mean))
    ax.annotate('Best: 12.8s', xy=(best_idx, total_mean[best_idx]/1000),
                xytext=(best_idx + 0.7, total_mean[best_idx]/1000 - 2),
                fontsize=11, fontweight='bold', color='#1B5E20',
                arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=2))

    ax.set_xlabel('Voice Assistant Version')
    ax.set_ylabel('Total Processing Latency (seconds)')
    ax.set_title('Total Latency Across All Versions (N=10 Trials, RPi4)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontweight='bold')
    ax.set_ylim(0, 20)

    fig.savefig(os.path.join(OUT_DIR, "fig1_total_latency.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig1_total_latency.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Stacked Component Breakdown (STT + LLM + TTS)
# ═══════════════════════════════════════════════════════════════════

def fig2_component_breakdown():
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(versions))
    width = 0.55

    stt_s = [v/1000 for v in stt_mean]
    llm_s = [v/1000 for v in llm_mean]
    tts_s = [v/1000 for v in tts_mean]

    p1 = ax.bar(x, stt_s, width, label='STT', color=C_STT, edgecolor='white')
    p2 = ax.bar(x, llm_s, width, bottom=stt_s, label='LLM', color=C_LLM, edgecolor='white')
    p3 = ax.bar(x, tts_s, width, bottom=[s+l for s,l in zip(stt_s, llm_s)],
                label='TTS', color=C_TTS, edgecolor='white')

    # Add component labels inside bars
    for i in range(len(versions)):
        # STT label
        if stt_s[i] > 1.0:
            ax.text(x[i], stt_s[i]/2, f'{stt_s[i]:.1f}s',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        # LLM label
        if llm_s[i] > 1.0:
            ax.text(x[i], stt_s[i] + llm_s[i]/2, f'{llm_s[i]:.1f}s',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)
        # TTS label
        if tts_s[i] > 1.0:
            ax.text(x[i], stt_s[i] + llm_s[i] + tts_s[i]/2, f'{tts_s[i]:.1f}s',
                    ha='center', va='center', color='white', fontweight='bold', fontsize=10)

    # Total labels on top
    for i, t in enumerate(total_mean):
        ax.text(x[i], t/1000 + 0.3, f'{t/1000:.1f}s',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_xlabel('Voice Assistant Version')
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Component Latency Breakdown: STT + LLM + TTS (N=10 Trials)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontweight='bold')
    ax.set_ylim(0, 18)
    ax.legend(loc='upper right', framealpha=0.9)

    # Bracket annotation for V1/V2
    ax.annotate('Whisper base\nFP32 PyTorch\n(11.8s STT!)', xy=(0.5, 12),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#BBDEFB', edgecolor=C_STT))

    fig.savefig(os.path.join(OUT_DIR, "fig2_component_breakdown.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig2_component_breakdown.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 3: STT Engine Comparison
# ═══════════════════════════════════════════════════════════════════

def fig3_stt_comparison():
    fig, ax = plt.subplots(figsize=(10, 5))

    configs = [
        "V1/V2\nWhisper base\nPyTorch FP32",
        "V3\nWhisper tiny\nPyTorch FP32",
        "V3-opt\nfaster-whisper\ntiny INT8",
        "V4\nfaster-whisper\nbase INT8"
    ]
    latencies = [11821, 4653, 2566, 4545]
    stds      = [56, 35, 45, 87]
    colors    = [C_RED, C_LLM, C_TTS, C_STT]

    x = np.arange(len(configs))
    bars = ax.bar(x, [l/1000 for l in latencies], 0.55,
                  yerr=[s/1000 for s in stds],
                  color=colors, edgecolor='white', linewidth=1.5,
                  capsize=6, error_kw={'linewidth': 2})

    for bar, l in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{l/1000:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=13)

    # Speedup annotations
    ax.annotate('', xy=(2, 2.566), xytext=(0, 11.821),
                arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=2.5, linestyle='--'))
    ax.text(1, 7.5, '4.6× faster', fontsize=12, fontweight='bold', color='#1B5E20',
            ha='center', rotation=-30)

    ax.set_ylabel('STT Latency (seconds)')
    ax.set_title('Speech-to-Text Engine Comparison on RPi4 (N=10 Trials)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylim(0, 14)

    fig.savefig(os.path.join(OUT_DIR, "fig3_stt_comparison.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig3_stt_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 4: V4 Per-Trial Breakdown (shows response-length variance)
# ═══════════════════════════════════════════════════════════════════

def fig4_v4_per_trial():
    fig, ax = plt.subplots(figsize=(14, 6))

    # Sort trials by total time for clarity
    order = np.argsort(v4_trials_total)

    x = np.arange(10)
    width = 0.6

    stt_sorted = [v4_trials_stt[i]/1000 for i in order]
    llm_sorted = [v4_trials_llm[i]/1000 for i in order]
    tts_sorted = [v4_trials_tts[i]/1000 for i in order]
    total_sorted = [v4_trials_total[i]/1000 for i in order]
    queries_sorted = [v4_trials_query[i] for i in order]

    p1 = ax.bar(x, stt_sorted, width, label='STT', color=C_STT, edgecolor='white')
    p2 = ax.bar(x, llm_sorted, width, bottom=stt_sorted, label='LLM', color=C_LLM, edgecolor='white')
    p3 = ax.bar(x, tts_sorted, width,
                bottom=[s+l for s,l in zip(stt_sorted, llm_sorted)],
                label='TTS', color=C_TTS, edgecolor='white')

    # Total labels
    for i, t in enumerate(total_sorted):
        ax.text(x[i], t + 0.3, f'{t:.1f}s', ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    # Query labels at bottom
    ax.set_xticks(x)
    ax.set_xticklabels(queries_sorted, fontsize=8, ha='center')

    # Bracket for short vs long queries
    ax.axhspan(0, 0, xmin=0, xmax=0.55, alpha=0)  # dummy
    ax.axvline(x=4.5, color=C_GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax.text(2, 28, 'Short responses (≤15 words)\nMean: 9.2s', fontsize=10,
            ha='center', color='#1B5E20', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#1B5E20'))
    ax.text(7.5, 28, 'Long responses (>50 words)\nMean: 25.4s', fontsize=10,
            ha='center', color='#B71C1C', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#FFEBEE', edgecolor='#B71C1C'))

    ax.set_ylabel('Latency (seconds)')
    ax.set_title('V4 Per-Trial Latency Breakdown — Response Length Drives Variance', fontweight='bold')
    ax.set_ylim(0, 32)
    ax.legend(loc='upper left', framealpha=0.9)

    fig.savefig(os.path.join(OUT_DIR, "fig4_v4_per_trial.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig4_v4_per_trial.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 5: YOLO Detection FPS Progression
# ═══════════════════════════════════════════════════════════════════

def fig5_yolo_fps():
    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = np.arange(len(yolo_phases))
    colors = ['#90CAF9', '#64B5F6', '#42A5F5', '#2196F3', C_RED, '#1976D2', '#0D47A1']

    bars = ax.bar(x, yolo_fps, 0.55, color=colors, edgecolor='white', linewidth=1.5)

    # Phase 5 special marker
    bars[4].set_hatch('///')
    bars[4].set_edgecolor(C_RED)

    # Value labels
    for bar, fps in zip(bars, yolo_fps):
        label = f'{fps} FPS' if fps > 0 else '0 FPS\n(FAILED)'
        color = 'white' if fps > 3 else C_RED
        y_pos = max(bar.get_height()/2, 1.0)
        ax.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                ha='center', va='center', fontweight='bold', fontsize=12, color=color)

    # 10× annotation
    ax.annotate('10× improvement', xy=(6, 20), xytext=(4, 21.5),
                fontsize=13, fontweight='bold', color='#0D47A1',
                arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=2.5))

    ax.set_xlabel('Optimization Phase')
    ax.set_ylabel('Frames Per Second')
    ax.set_title('YOLO Detection: FPS Progression Across 7 Optimization Phases', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(yolo_phases, fontsize=9)
    ax.set_ylim(0, 24)

    fig.savefig(os.path.join(OUT_DIR, "fig5_yolo_fps.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig5_yolo_fps.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 6: Optimization Impact Hierarchy (Both Projects)
# ═══════════════════════════════════════════════════════════════════

def fig6_optimization_impact():
    fig, ax = plt.subplots(figsize=(10, 5))

    categories = [
        'Runtime / Backend\nSelection',
        'Resolution &\nArchitecture',
        'Model\nQuantization',
        'Pipeline\nEngineering',
        'Code-Level\nOptimization'
    ]
    impacts = [48, 21, 16, 10, 5]
    colors = ['#1565C0', '#1976D2', '#2196F3', '#64B5F6', '#BBDEFB']

    bars = ax.barh(np.arange(len(categories)), impacts, 0.55,
                   color=colors, edgecolor='white', linewidth=1.5)

    for bar, imp in zip(bars, impacts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{imp}%', ha='left', va='center', fontweight='bold', fontsize=13)

    # Example annotations
    ax.text(24, 0, 'PyTorch→TFLite, PyTorch→CTranslate2', fontsize=9, va='center', color=C_GRAY)
    ax.text(11, 1, '320→224px', fontsize=9, va='center', color=C_GRAY)
    ax.text(8, 2, 'FP32→INT8, q4→q2', fontsize=9, va='center', color=C_GRAY)

    ax.set_xlabel('Share of Total Performance Gain (%)')
    ax.set_title('Optimization Impact Hierarchy — Both Projects Combined', fontweight='bold')
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlim(0, 60)
    ax.invert_yaxis()

    fig.savefig(os.path.join(OUT_DIR, "fig6_optimization_impact.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig6_optimization_impact.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 7: Capability vs Latency (Key Insight!)
# ═══════════════════════════════════════════════════════════════════

def fig7_capability_vs_latency():
    """The KEY graph: shows that total latency stays ~13-14s while capability jumps."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(versions))

    # Capability score (0–10 scale)
    # V1/V2: keyword matching = 2, V3: LLM 4-bit = 7, V3-opt: LLM 2-bit + faster-whisper = 8, V4: conversational AI = 9
    capability = [2, 2, 7, 8, 9]

    # Bar for total latency
    bars = ax1.bar(x, [t/1000 for t in total_mean], 0.45,
                   color=[C_GRAY, C_GRAY, '#CE93D8', '#AB47BC', C_TOTAL],
                   edgecolor='white', linewidth=1.5,
                   yerr=[s/1000 for s in total_std], capsize=5,
                   error_kw={'linewidth': 1.5}, alpha=0.8, label='Total Latency')
    ax1.set_ylabel('Total Latency (seconds)', color=C_TOTAL)
    ax1.set_ylim(0, 22)

    # Line for capability
    ax2 = ax1.twinx()
    ax2.plot(x, capability, 'o-', color='#E65100', linewidth=3, markersize=12,
             markerfacecolor='#FFB74D', markeredgecolor='#E65100', markeredgewidth=2,
             label='Capability Level', zorder=5)
    ax2.set_ylabel('AI Capability Level', color='#E65100')
    ax2.set_ylim(0, 12)

    # Capability labels
    cap_labels = ['Keyword\nMatching', 'Keyword\nMatching', 'LLM\n(4-bit)', 'LLM\n(2-bit)', 'Full\nConversational AI']
    for i, (c, lbl) in enumerate(zip(capability, cap_labels)):
        ax2.text(i, c + 0.6, lbl, ha='center', va='bottom', fontsize=8,
                 color='#E65100', fontweight='bold')

    # Highlight the key insight
    ax1.axhline(y=13.5, color=C_GRAY, linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(4.5, 13.8, 'All versions: ~13–14s total', fontsize=10, ha='right',
             color=C_GRAY, style='italic')

    ax1.set_xlabel('Voice Assistant Version')
    ax1.set_title('Key Insight: Latency Stays Constant While Capability Increases', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(versions, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

    fig.savefig(os.path.join(OUT_DIR, "fig7_capability_vs_latency.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig7_capability_vs_latency.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 8: INT8 Quantization Failure Visualization
# ═══════════════════════════════════════════════════════════════════

def fig8_int8_failure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left panel: value ranges
    categories = ['Bounding Box\n(x, y, w, h)', 'Objectness\nScore', 'Class\nProbabilities']
    expected = [320, 1.0, 1.0]
    actual = [1.47, 0.08, 0.08]

    x = np.arange(3)
    w = 0.35
    ax1.bar(x - w/2, expected, w, label='Expected Range', color=C_STT, edgecolor='white')
    ax1.bar(x + w/2, actual, w, label='After INT8 Quantization', color=C_RED, edgecolor='white')

    # Scale bars differently - use log scale
    ax1.set_yscale('log')
    ax1.set_ylim(0.01, 500)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.set_ylabel('Value Range (log scale)')
    ax1.set_title('Value Range Compression\n(Per-Tensor INT8)', fontweight='bold')
    ax1.legend()

    # Right panel: confidence impact
    thresh = [0.25, 0.5]
    max_conf = 0.0064

    ax2.barh([0], [max_conf], 0.4, color=C_RED, edgecolor='white', label=f'Max achievable: {max_conf}')
    for i, t in enumerate(thresh):
        ax2.axvline(x=t, color=['#FFA726', '#E65100'][i], linestyle='--', linewidth=2,
                    label=f'Threshold: {t}')

    ax2.set_xlim(0, 0.7)
    ax2.set_yticks([0])
    ax2.set_yticklabels(['Confidence\n(obj × class)'])
    ax2.set_xlabel('Confidence Score')
    ax2.set_title('INT8 Max Confidence vs.\nDetection Threshold', fontweight='bold')
    ax2.legend(fontsize=10)

    # Big red X
    ax2.text(0.35, 0.55, '✗ ALL DETECTIONS LOST', fontsize=14, fontweight='bold',
             color=C_RED, transform=ax2.transAxes, ha='center')

    fig.suptitle('INT8 Per-Tensor Quantization Failure in YOLO Multi-Head Output',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig8_int8_failure.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig8_int8_failure.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 9: STT Consistency vs LLM/TTS Variance
# ═══════════════════════════════════════════════════════════════════

def fig9_variance_analysis():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Coefficient of variation (%) for each component across versions with LLM
    cv_labels = ['V3', 'V3-opt', 'V4']
    cv_stt = [35/4653*100, 45/2566*100, 87/4545*100]
    cv_llm = [4579/5299*100, 3953/5464*100, 2509/4337*100]
    cv_tts = [1415/4029*100, 1316/4726*100, 608/4144*100]

    x = np.arange(3)
    w = 0.25

    ax.bar(x - w, cv_stt, w, label='STT (deterministic)', color=C_STT, edgecolor='white')
    ax.bar(x, cv_llm, w, label='LLM (response-length dependent)', color=C_LLM, edgecolor='white')
    ax.bar(x + w, cv_tts, w, label='TTS (response-length dependent)', color=C_TTS, edgecolor='white')

    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('Latency Variance by Component: STT is Deterministic, LLM/TTS Scale with Output',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cv_labels, fontweight='bold')
    ax.legend(loc='upper right')

    # Annotate key insight
    ax.text(1, 80, 'STT processes fixed-length audio → consistent\n'
                    'LLM/TTS produce variable-length output → high variance',
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='#FFF9C4', edgecolor='#F9A825'))

    fig.savefig(os.path.join(OUT_DIR, "fig9_variance_analysis.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig9_variance_analysis.png")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 10: Summary Dashboard (2-panel overview)
# ═══════════════════════════════════════════════════════════════════

def fig10_summary():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: YOLO 2→20 FPS
    ax1.bar(['Baseline\n(PyTorch)'], [2], color=C_RED, edgecolor='white', width=0.4)
    ax1.bar(['Final\n(TFLite FP16)'], [20], color='#0D47A1', edgecolor='white', width=0.4)
    ax1.set_ylabel('Frames Per Second')
    ax1.set_title('Part A: YOLO Detection\n10× Throughput Improvement', fontweight='bold')
    ax1.set_ylim(0, 25)
    for i, (v, c) in enumerate(zip([2, 20], [C_RED, '#0D47A1'])):
        ax1.text(i, v + 0.5, f'{v} FPS', ha='center', fontweight='bold', fontsize=16, color=c)
    ax1.annotate('10×', xy=(0.5, 11), fontsize=28, fontweight='bold',
                 color='#1B5E20', ha='center',
                 xycoords=('axes fraction', 'data'))

    # Right: Voice — stacked bar for V4 with annotations
    bottom = 0
    comps = [('STT\n4.5s', 4545/1000, C_STT),
             ('LLM\n4.3s', 4337/1000, C_LLM),
             ('TTS\n4.1s', 4144/1000, C_TTS)]
    for label, val, color in comps:
        ax2.bar(['V4 Pipeline'], [val], bottom=[bottom], color=color, edgecolor='white', width=0.4)
        ax2.text(0, bottom + val/2, label, ha='center', va='center',
                 color='white', fontweight='bold', fontsize=11)
        bottom += val

    ax2.text(0, bottom + 0.3, f'{sum(c[1] for c in comps):.1f}s total',
             ha='center', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Processing Time (seconds)')
    ax2.set_title('Part B: Voice Assistant (V4)\nFully Offline Conversational AI', fontweight='bold')
    ax2.set_ylim(0, 18)

    # Key facts
    ax2.text(0.95, 0.95, '• Fully offline\n• No cloud dependency\n• RPi4, 4GB RAM\n• $35 hardware',
             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#388E3C'))

    fig.suptitle('Real-Time AI on Raspberry Pi 4 — Key Results', fontweight='bold', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "fig10_summary_dashboard.png"), bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig10_summary_dashboard.png")


# ═══════════════════════════════════════════════════════════════════
# Generate all figures
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\nGenerating paper figures → {OUT_DIR}/\n")

    fig1_total_latency()
    fig2_component_breakdown()
    fig3_stt_comparison()
    fig4_v4_per_trial()
    fig5_yolo_fps()
    fig6_optimization_impact()
    fig7_capability_vs_latency()
    fig8_int8_failure()
    fig9_variance_analysis()
    fig10_summary()

    print(f"\n✅ All 10 figures generated in {OUT_DIR}/")
    print("\nFigures:")
    print("  1. Total latency comparison (all 5 versions)")
    print("  2. Component breakdown (STT + LLM + TTS stacked)")
    print("  3. STT engine comparison (4 configs)")
    print("  4. V4 per-trial breakdown (response-length variance)")
    print("  5. YOLO FPS progression (7 phases)")
    print("  6. Optimization impact hierarchy (horizontal bar)")
    print("  7. KEY INSIGHT: Capability vs Latency (dual axis)")
    print("  8. INT8 quantization failure visualization")
    print("  9. Variance analysis (CV% by component)")
    print(" 10. Summary dashboard (2-panel overview)")

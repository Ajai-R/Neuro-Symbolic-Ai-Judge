"""
Generates publication-quality figures for the IEEE Conference Research Paper.
Figures produced:
1. fig_confusion_matrix.png (300 DPI)
2. fig_ablation_f1.png (300 DPI)
3. fig_coverage_matrix.png (300 DPI)
4. Copies Dataset/arch_diagram.png to figures/fig_architecture.png
"""

import os
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Target directory
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Set global styles suitable for IEEE publications
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

CLASSES = [
    "INFO_DIRECTED",
    "APPEAL_DISMISSED",
    "PENALTY_IMPOSED",
    "PARTIAL_RELIEF"
]

FULL_CLASSES = [
    "Information Directed",
    "Appeal Dismissed",
    "Penalty Imposed",
    "Partial Relief"
]

# -----------------------------------------------------------------------------
# 1. Figure 1: Confusion Matrix of Master Decision Engine (N=424 Blind Test)
# -----------------------------------------------------------------------------
cm = np.array([
    [69, 34,  3,  0],   # INFORMATION_DIRECTED
    [21, 81,  3,  1],   # APPEAL_DISMISSED
    [11,  5, 88,  2],   # PENALTY_IMPOSED
    [10, 13,  3, 80]    # PARTIAL_RELIEF
])

fig, ax = plt.subplots(figsize=(4.8, 4.0), dpi=300)
cax = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Master Engine Test Confusion Matrix (N=424)", pad=12, fontweight='bold')
cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=8)

tick_marks = np.arange(len(CLASSES))
ax.set_xticks(tick_marks)
ax.set_yticks(tick_marks)
ax.set_xticklabels(CLASSES, rotation=30, ha='right', fontsize=8, fontweight='medium')
ax.set_yticklabels(CLASSES, fontsize=8, fontweight='medium')

# Annotate each cell with count and percentage
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        pct = (val / 106.0) * 100
        text_color = "white" if val > thresh else "black"
        ax.text(j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center", color=text_color, fontsize=8.5, fontweight='bold')

ax.set_ylabel("True Case Outcome", fontweight='bold', labelpad=6)
ax.set_xlabel("Predicted Advisory Verdict", fontweight='bold', labelpad=6)
plt.tight_layout()
cm_path = os.path.join(FIG_DIR, "fig_confusion_matrix.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Generated: {cm_path}")

# -----------------------------------------------------------------------------
# 2. Figure 2: Ablation Study Comparison (Per-class & Macro F1)
# -----------------------------------------------------------------------------
# Comparison between:
# (a) Signal 1 (InLegalBERT Prior alone)
# (b) Multi-Signal Weighted Fusion (Signals 1 + 2 + 3)
# (c) Master Neuro-Symbolic Engine (Tri-Signal + Hierarchical Guardrails)

categories = ["Info Directed", "Dismissed", "Penalty", "Partial Relief", "Macro F1"]
s1_scores = [66.9, 56.7, 86.1, 83.7, 73.4]       # Signal 1
fusion_scores = [64.9, 65.8, 86.0, 84.7, 75.3]   # Weighted Fusion (no guardrail overrides)
master_scores = [63.6, 67.8, 86.7, 84.7, 75.7]   # Master Engine (with statutory guardrail overrides)

x = np.arange(len(categories)) * 1.5
width = 0.29

fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)
rects1 = ax.bar(x - width, s1_scores, width, label='Signal 1 (InLegalBERT alone)', color='#4C72B0', edgecolor='#222222', linewidth=0.8)
rects2 = ax.bar(x, fusion_scores, width, label='Weighted Fusion (w/o Guardrails)', color='#55A868', edgecolor='#222222', linewidth=0.8)
rects3 = ax.bar(x + width, master_scores, width, label='Master Neuro-Symbolic (Full)', color='#C44E52', edgecolor='#222222', linewidth=0.8)

ax.set_ylabel('F1-Score (%)', fontweight='bold', fontsize=11)
ax.set_title('Ablation Study: Neural Prior vs. Fusion vs. Guardrails', pad=14, fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10.5, fontweight='bold')
ax.set_ylim(40, 105)
ax.yaxis.grid(True, linestyle='--', alpha=0.55)
ax.set_axisbelow(True)

# Place legend in upper left where bars are low (<= 67.8%) to leave high bars completely unobstructed
ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper left', fontsize=9.0, edgecolor='#cccccc')

# Add value labels above bars with clean bold typography
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3.5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, fontweight='bold', color='#111111')

plt.tight_layout()
ablation_path = os.path.join(FIG_DIR, "fig_ablation_f1.png")
plt.savefig(ablation_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"[OK] Generated: {ablation_path}")

# -----------------------------------------------------------------------------
# 3. Figure 3: Dense Semantic Alignment Matrix Heatmap (Signal 2 Coverage)
# -----------------------------------------------------------------------------
# Illustrates the N x M cosine similarity alignment for a multi-point dispute:
# Q1: Certified copies of tender bidding documents
# Q2: Final sanction order and financial expenditure approval
# Q3: Inspection report submitted by vigilance committee
# Q4: List of official attendees at tender board meeting

q_labels = [
    "Q1: Tender Bid Documents",
    "Q2: Sanction Order & Approval",
    "Q3: Vigilance Inspection Report",
    "Q4: Meeting Attendance List"
]

r_labels = [
    "R1: Furnished tender bids",
    "R2: Provided sanction copies",
    "R3: Vigilance inquiry pending",
    "R4: Routine log maintained",
    "R5: Fee receipt acknowledged"
]

# -----------------------------------------------------------------------------
# 3. Figure 2: Signal 2 LLM-as-a-Judge Substantive Compliance Ledger
# -----------------------------------------------------------------------------
from generate_coverage_ledger import create_compliance_ledger
cov_path = os.path.join(FIG_DIR, "fig_coverage_matrix.png")
create_compliance_ledger(cov_path)
print(f"[OK] Generated new Signal 2 Compliance Ledger: {cov_path}")

# -----------------------------------------------------------------------------
# 4. Copy Architecture Diagram
# -----------------------------------------------------------------------------
arch_src = os.path.join(os.path.dirname(FIG_DIR), "Dataset", "arch_diagram.png")
arch_dst = os.path.join(FIG_DIR, "fig_architecture.png")
if os.path.exists(arch_src):
    shutil.copyfile(arch_src, arch_dst)
    print(f"[OK] Copied architecture diagram to: {arch_dst}")
else:
    print(f"[WARNING] arch_src not found at {arch_src}")

print("\nAll figures generated successfully.")

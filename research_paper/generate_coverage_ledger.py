"""
Generates publication-quality Figure 2 for the IEEE Conference Research Paper:
"Signal 2: Itemized Question-Response Substantive Compliance Ledger (LLM-as-a-Judge)"
Replaces the outdated BGE-M3 cosine similarity matrix with the new LLM-as-a-Judge
factual audit matrix.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def create_compliance_ledger(output_path):
    fig_w, fig_h = 7.0, 2.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    font_main = 'Segoe UI'

    # Background card
    bg_box = FancyBboxPatch((0.01, 0.01), 0.98, 0.98,
                            boxstyle="round,pad=0.008,rounding_size=0.015",
                            facecolor='#FAFAFA', edgecolor='#CBD5E1', lw=1.2, zorder=1)
    ax.add_patch(bg_box)

    # Title Banner
    ax.text(0.50, 0.940, "Signal 2: LLM-as-a-Judge Substantive Compliance Ledger",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E293B', family=font_main, zorder=2)
    ax.text(0.50, 0.880, "Itemized Question-Response Factual Audit Matrix & Dynamic Statutory Weighting",
            ha='center', va='center', fontsize=7.2, fontstyle='italic', color='#64748B', family=font_main, zorder=2)

    # Table Header Row
    header_y = 0.800
    header_box = FancyBboxPatch((0.025, header_y - 0.032), 0.950, 0.058,
                               boxstyle="round,pad=0.004,rounding_size=0.008",
                               facecolor='#E2E8F0', edgecolor='#94A3B8', lw=0.9, zorder=2)
    ax.add_patch(header_box)

    col_x = [0.040, 0.325, 0.470, 0.585]
    ax.text(col_x[0], header_y, r"Citizen Demand ($Q_i$)", ha='left', va='center', fontsize=7.2, fontweight='bold', color='#1E293B', family=font_main, zorder=3)
    ax.text(col_x[1], header_y, r"Weight ($\omega_i$)", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1E293B', family=font_main, zorder=3)
    ax.text(col_x[2], header_y, r"LLM Status ($s_i$)", ha='center', va='center', fontsize=7.2, fontweight='bold', color='#1E293B', family=font_main, zorder=3)
    ax.text(col_x[3], header_y, r"Evidentiary PIO Excerpt / Grounded Finding", ha='left', va='center', fontsize=7.2, fontweight='bold', color='#1E293B', family=font_main, zorder=3)

    # Rows Data
    rows = [
        {
            "query": r"$Q_1$: Certified Tender Bids",
            "weight": r"Core ($\omega_1 = 0.65$)",
            "status": "FURNISHED",
            "score": r"$s_1 = 1.0$",
            "theme": {"bg": "#DCFCE7", "border": "#16A34A", "text": "#15803D"},
            "finding": '"CPIO supplied certified copies of bid sheets in hearing"'
        },
        {
            "query": r"$Q_2$: Expenditure Sanctions",
            "weight": r"Core ($\omega_2 = 0.65$)",
            "status": "FURNISHED",
            "score": r"$s_2 = 1.0$",
            "theme": {"bg": "#DCFCE7", "border": "#16A34A", "text": "#15803D"},
            "finding": '"Sanction order copy No. 402 furnished on official record"'
        },
        {
            "query": r"$Q_3$: Vigilance Inquiry Report",
            "weight": r"Core ($\omega_3 = 0.65$)",
            "status": "WITHHELD_EXEMPT",
            "score": r"$s_3 = 0.0$",
            "theme": {"bg": "#FEF3C7", "border": "#D97706", "text": "#B45309"},
            "finding": '"Withheld under Sec. 8(1)(h): active ongoing inquiry"'
        },
        {
            "query": r"$Q_4$: Meeting Attendance Log",
            "weight": r"Aux ($\omega_4 = 0.35$)",
            "status": "EVADED_OMITTED",
            "score": r"$s_4 = 0.0$",
            "theme": {"bg": "#FEE2E2", "border": "#DC2626", "text": "#B91C1C"},
            "finding": '"Omitted in CPIO written rejoinder; no defense cited"'
        }
    ]

    curr_y = 0.680
    row_height = 0.100

    for i, r in enumerate(rows):
        # Row background
        r_box = FancyBboxPatch((0.025, curr_y - 0.046), 0.950, 0.092,
                               boxstyle="round,pad=0.004,rounding_size=0.008",
                               facecolor='#FFFFFF', edgecolor='#E2E8F0', lw=0.8, zorder=2)
        ax.add_patch(r_box)

        # Query
        ax.text(col_x[0], curr_y, r["query"], ha='left', va='center', fontsize=7.1, fontweight='bold', color='#1E293B', family=font_main, zorder=3)
        # Weight
        ax.text(col_x[1], curr_y, r["weight"], ha='center', va='center', fontsize=6.8, color='#475569', family=font_main, zorder=3)
        
        # Status Badge
        badge_w, badge_h = 0.145, 0.062
        badge_x = col_x[2] - badge_w/2.0
        badge_y = curr_y - badge_h/2.0
        badge = FancyBboxPatch((badge_x, badge_y), badge_w, badge_h,
                               boxstyle="round,pad=0.003,rounding_size=0.008",
                               facecolor=r["theme"]["bg"], edgecolor=r["theme"]["border"], lw=1.0, zorder=3)
        ax.add_patch(badge)
        ax.text(col_x[2], curr_y + 0.012, r["status"], ha='center', va='center', fontsize=5.6, fontweight='bold', color=r["theme"]["text"], family=font_main, zorder=4)
        ax.text(col_x[2], curr_y - 0.014, r["score"], ha='center', va='center', fontsize=6.2, color=r["theme"]["text"], family=font_main, zorder=4)

        # Finding excerpt
        ax.text(col_x[3], curr_y, r["finding"], ha='left', va='center', fontsize=6.5, fontstyle='italic', color='#334155', family=font_main, zorder=3)

        curr_y -= 0.105

    # Bottom Summary Metric Card
    summary_y = 0.135
    sum_box = FancyBboxPatch((0.025, 0.030), 0.950, 0.190,
                             boxstyle="round,pad=0.005,rounding_size=0.010",
                             facecolor='#EFF6FF', edgecolor='#3B82F6', lw=1.1, zorder=2)
    ax.add_patch(sum_box)

    formula_text = r"Aggregate Coverage: $\rho = \frac{\sum \omega_i \cdot s_i}{\sum \omega_i} = \frac{0.65(1.0) + 0.65(1.0) + 0.65(0.0) + 0.35(0.0)}{0.65 + 0.65 + 0.65 + 0.35} = \frac{1.30}{2.30} = \mathbf{0.565}$"
    ax.text(0.040, 0.155, formula_text, ha='left', va='center', fontsize=6.8, color='#1E3A8A', family=font_main, zorder=3)

    verdict_text = r"LLM Factual Audit Finding: Coverage Ratio $\rho = 0.565$ ($0.20 \leq \rho \leq 0.80$) $\to$ PARTIAL COMPLIANCE $\Rightarrow$ Enforces Sec. 10 Severability"
    ax.text(0.040, 0.075, verdict_text, ha='left', va='center', fontsize=7.0, fontweight='bold', color='#1D4ED8', family=font_main, zorder=3)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"[SUCCESS] Signal 2 Compliance Ledger written to: {output_path}")

if __name__ == '__main__':
    out_path = 'research_paper/figures/fig_coverage_matrix.png'
    create_compliance_ledger(out_path)

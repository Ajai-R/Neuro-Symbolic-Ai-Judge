"""
Generates a human-grade, clean, publication-quality boxes-and-lines diagram for:
Signal 2: LLM-as-a-Judge Substantive Coverage Verification Engine
Style: Clean academic boxes, directional arrows, badges, zero AI artifacts.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_signal2_diagram(output_path):
    # Width: 12.0 inches, Height: 5.2 inches (Aspect ratio ~ 2.3:1, perfect for column width)
    fig, ax = plt.subplots(figsize=(12.0, 5.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Color Palette - Professional Academic Theme
    C_INPUT  = {'border': '#1E3A8A', 'bg': '#EFF6FF', 'title': '#1E3A8A', 'text': '#1E293B'}
    C_PROC   = {'border': '#B45309', 'bg': '#FFFBEB', 'title': '#92400E', 'text': '#78350F'}
    C_AUDIT  = {'border': '#0F766E', 'bg': '#F0FDFA', 'title': '#115E59', 'text': '#134E4A'}
    C_AGG    = {'border': '#4338CA', 'bg': '#EEF2FF', 'title': '#3730A3', 'text': '#312E81'}
    C_OUT    = {'border': '#15803D', 'bg': '#F0FDF4', 'title': '#166534', 'text': '#14532D'}

    font_family = 'DejaVu Sans'

    def draw_box(x, y, w, h, c, title, items, badge="", title_size=9.5, item_size=8.0):
        # Outer box
        rect = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.008,rounding_size=0.015",
                              facecolor=c['bg'], edgecolor=c['border'],
                              linewidth=1.6, zorder=2)
        ax.add_patch(rect)

        # Optional Top Badge
        if badge:
            bw, bh = len(badge) * 0.0095 + 0.04, 0.038
            bx = x + w/2 - bw/2
            by = y + h - bh/2
            badge_box = FancyBboxPatch((bx, by), bw, bh,
                                       boxstyle="round,pad=0.004,rounding_size=0.01",
                                       facecolor=c['border'], edgecolor='none', zorder=4)
            ax.add_patch(badge_box)
            ax.text(x + w/2, by + bh/2, badge, ha='center', va='center',
                    color='white', fontsize=7.2, fontweight='bold', family=font_family, zorder=5)

        # Title
        ty = y + h - 0.048 if badge else y + h - 0.038
        ax.text(x + w/2, ty, title, ha='center', va='center',
                color=c['title'], fontsize=title_size, fontweight='bold', family=font_family, zorder=3)

        # Divider line
        div_y = ty - 0.024
        ax.plot([x + 0.012, x + w - 0.012], [div_y, div_y],
                color=c['border'], linewidth=0.8, linestyle='-', alpha=0.45, zorder=3)

        # Items
        cur_y = div_y - 0.032
        for it in items:
            ax.text(x + 0.016, cur_y, it, ha='left', va='center',
                    color=c['text'], fontsize=item_size, family=font_family, zorder=3)
            cur_y -= 0.038

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#334155", lw=1.6,
                                    mutation_scale=12),
                    zorder=4)
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2 + 0.026
            ax.text(mx, my, label, ha='center', va='center',
                    fontsize=7.2, fontweight='bold', color="#1E293B",
                    family=font_family,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", lw=0.8),
                    zorder=5)

    # -------------------------------------------------------------
    # 1. Inputs (Left)
    # -------------------------------------------------------------
    draw_box(0.02, 0.15, 0.17, 0.72, C_INPUT,
             "Input Case Dossier",
             [
                 r"• $\mathbf{x}_{\mathrm{query}}$: Requisition",
                 "  Discrete demands",
                 r"  $\mathcal{Q} = \{Q_1, \dots, Q_N\}$",
                 "",
                 r"• $\mathbf{x}_{\mathrm{narrative}}$: Statements",
                 "  CPIO written reply",
                 "  Procedural facts",
                 "  Hearing submissions"
             ],
             badge="STAGE 1: INPUT")

    # Arrow 1 -> 2
    draw_arrow(0.19, 0.51, 0.22, 0.51, "Tokenize")

    # -------------------------------------------------------------
    # 2. Query Decomposer & Dynamic Weighting
    # -------------------------------------------------------------
    draw_box(0.22, 0.15, 0.22, 0.72, C_PROC,
             "Demand Itemization",
             [
                 r"• Clause Partitioning:",
                 r"  $\mathcal{Q} \to [Q_1, Q_2, \dots, Q_N]$",
                 "",
                 r"• Dynamic Weights $\omega_i \in [0.35, 0.65]$:",
                 r"  - Core Records ($\omega_i \approx 0.65$):",
                 "    Sanctions, accounts, files",
                 r"  - Auxiliary Info ($\omega_i \approx 0.35$):",
                 "    Dispatch nos, dates"
             ],
             badge="STAGE 2: WEIGHTING")

    # Arrow 2 -> 3
    draw_arrow(0.44, 0.51, 0.47, 0.51, r"$\mathcal{Q}, \vec{\omega}$")

    # -------------------------------------------------------------
    # 3. LLM-as-a-Judge Substantive Compliance Auditor
    # -------------------------------------------------------------
    box3 = FancyBboxPatch((0.47, 0.10), 0.26, 0.82,
                          boxstyle="round,pad=0.008,rounding_size=0.015",
                          facecolor=C_AUDIT['bg'], edgecolor=C_AUDIT['border'],
                          linewidth=1.6, zorder=2)
    ax.add_patch(box3)

    # Badge
    ax.text(0.60, 0.92, "STAGE 3: LLM JUDICIAL AUDITOR", ha='center', va='center',
            color='white', fontsize=7.2, fontweight='bold', family=font_family, zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc=C_AUDIT['border'], ec='none'))

    ax.text(0.60, 0.865, "LLM-as-a-Judge Compliance Engine", ha='center', va='center',
            color=C_AUDIT['title'], fontsize=9.5, fontweight='bold', family=font_family, zorder=3)
    ax.plot([0.485, 0.715], [0.835, 0.835], color=C_AUDIT['border'], lw=0.8, alpha=0.45)

    # 4 Status Chips inside
    chips = [
        ("FURNISHED", r"$s_i = 1.0$", "#DCFCE7", "#166534", "Records specifically provided"),
        ("PARTIALLY FURNISHED", r"$s_i = 0.5$", "#FEF3C7", "#92400E", "Incomplete or redacted data"),
        ("WITHHELD (EXEMPT)", r"$s_i = 0.0$", "#FEE2E2", "#991B1B", "Legitimate Sec. 8/24 defense"),
        ("EVADED / OMITTED", r"$s_i = 0.0$", "#F3F4F6", "#374151", "Unlawful bureaucratic silence")
    ]
    chip_y = 0.77
    for title, score, bg, fg, desc in chips:
        ch_box = FancyBboxPatch((0.485, chip_y - 0.095), 0.23, 0.105,
                                boxstyle="round,pad=0.004,rounding_size=0.01",
                                facecolor=bg, edgecolor=fg, lw=1.1, zorder=3)
        ax.add_patch(ch_box)
        ax.text(0.495, chip_y - 0.030, title, ha='left', va='center',
                fontsize=7.8, fontweight='bold', color=fg, family=font_family, zorder=4)
        ax.text(0.705, chip_y - 0.030, score, ha='right', va='center',
                fontsize=7.8, fontweight='bold', color=fg, family=font_family, zorder=4)
        ax.text(0.495, chip_y - 0.068, desc, ha='left', va='center',
                fontsize=6.8, color="#1E293B", family=font_family, zorder=4)
        chip_y -= 0.125

    # Excerpt extraction note
    ax.text(0.60, 0.135, r"• Extracts verbatim evidentiary excerpts",
            ha='center', va='center', fontsize=7.2, color=C_AUDIT['text'], family=font_family, zorder=3)

    # Arrow 3 -> 4
    draw_arrow(0.73, 0.51, 0.76, 0.51, r"Scores $s_i$")

    # -------------------------------------------------------------
    # 4. Coverage Score & Output Vector (Right)
    # -------------------------------------------------------------
    draw_box(0.76, 0.12, 0.22, 0.78, C_OUT,
             "Factual Coverage & Output",
             [
                 r"• Coverage Ratio $\rho$:",
                 r"  $\rho = \frac{\sum_{i=1}^N \omega_i \cdot s_i}{\sum_{i=1}^N \omega_i}$",
                 "",
                 r"• Compliance Thresholds:",
                 r"  $\rho \geq 0.85 \to \mathrm{Complete}$",
                 r"  $0.30 \leq \rho < 0.85 \to \mathrm{Partial}$",
                 r"  $\rho < 0.30 \to \mathrm{Non\text{-}Compliant}$",
                 "",
                 r"• Outputs Synthesized:",
                 r"  - $\mathbf{p}_{S2}$: Probability Vector",
                 "  - Judicial Reasoning Trace",
                 "  - Section 10 Severability"
             ],
             badge="STAGE 4: ADVISORY OUTPUT",
             item_size=7.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Generated Signal 2 Boxes-and-Lines Architecture: {output_path}")

if __name__ == "__main__":
    out_file = r"d:\Final Year Project\research_paper\figures\fig_coverage_matrix.png"
    create_signal2_diagram(out_file)

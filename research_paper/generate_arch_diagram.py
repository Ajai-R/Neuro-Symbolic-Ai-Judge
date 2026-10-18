"""
Generates a publication-quality, human-grade vector architecture diagram for:
"Explainable Neuro-Symbolic AI for Judicial Outcome Prediction under the RTI Act"
Updates the methodology to the new architecture:
- Signal 2: LLM-as-a-Judge Substantive Coverage Verification Engine
- Accurate override rate (3.54%)
- Crystal-clear typography, professional academic color scheme, crisp mathematical notation.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_architecture_diagram(output_path):
    fig_w, fig_h = 16.0, 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Color Palette - Professional Academic IEEE Theme
    colors = {
        'input': {'border': '#475569', 'bg': '#F8FAFC', 'title': '#1E293B', 'text': '#334155'},
        's1':    {'border': '#16A34A', 'bg': '#F0FDF4', 'title': '#15803D', 'text': '#166534'},
        's2':    {'border': '#D97706', 'bg': '#FFFBEB', 'title': '#B45309', 'text': '#92400E'},
        's3':    {'border': '#9333EA', 'bg': '#FAF5FF', 'title': '#7E22CE', 'text': '#6B21A8'},
        'fused': {'border': '#0284C7', 'bg': '#F0F9FF', 'title': '#0369A1', 'text': '#075985'},
        'guard': {'border': '#DC2626', 'bg': '#FEF2F2', 'title': '#B91C1C', 'text': '#991B1B'},
        'output':{'border': '#4F46E5', 'bg': '#EEF2FF', 'title': '#4338CA', 'text': '#3730A3'},
        'arrow': '#334155',
        'guard_arrow': '#B91C1C'
    }

    # Font properties
    font_main = 'Segoe UI'
    
    def draw_card(x, y, w, h, theme, title, items, title_size=10.5, item_size=8.0, line_spacing=0.038, has_divider=True):
        # Background box
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.008,rounding_size=0.015",
                             facecolor=theme['bg'],
                             edgecolor=theme['border'],
                             linewidth=1.8,
                             zorder=2)
        ax.add_patch(box)
        
        # Title
        title_y = y + h - 0.040
        ax.text(x + w/2.0, title_y, title,
                ha='center', va='center',
                fontsize=title_size, fontweight='bold',
                color=theme['title'], family=font_main, zorder=3)
        
        # Divider
        if has_divider:
            div_y = title_y - 0.024
            ax.plot([x + 0.015, x + w - 0.015], [div_y, div_y],
                    color=theme['border'], linewidth=0.9, alpha=0.6, zorder=3)
            start_y = div_y - 0.030
        else:
            start_y = title_y - 0.035
            
        # Items
        curr_y = start_y
        for it in items:
            if it == '---':
                ax.plot([x + 0.02, x + w - 0.02], [curr_y + 0.01, curr_y + 0.01],
                        color=theme['border'], linestyle='--', linewidth=0.8, alpha=0.7, zorder=3)
                curr_y -= line_spacing * 0.7
                continue
                
            is_bold = it.startswith('**') and it.endswith('**')
            txt = it.replace('**', '') if is_bold else it
            fweight = 'bold' if is_bold else 'normal'
            fcolor = theme['title'] if is_bold else theme['text']
            
            # If item contains colon or sub-bullet
            ax.text(x + w/2.0, curr_y, txt,
                    ha='center', va='center',
                    fontsize=item_size, fontweight=fweight,
                    color=fcolor, family=font_main, zorder=3)
            curr_y -= line_spacing

    # =========================================================================
    # 1. Dispute Input Tuple D
    # =========================================================================
    draw_card(
        x=0.020, y=0.22, w=0.145, h=0.56,
        theme=colors['input'],
        title=r'Dispute Input Tuple $\mathcal{D}$',
        items=[
            r'Citizen Query Items: $\mathbf{x}_{\mathrm{query}}$',
            '',
            r'Hearing Record: $\mathbf{x}_{\mathrm{narrative}}$',
            '',
            r'Public Authority: $\mathbf{x}_{\mathrm{auth}}$',
            '',
            r'Exemption Claims: $\mathbf{x}_{\mathrm{exempt}}$',
            '',
            r'Procedural Timeline ($\Delta t$)'
        ],
        item_size=8.2, line_spacing=0.052
    )

    # =========================================================================
    # 2. Tri-Signal Extraction (Signals 1, 2, 3)
    # =========================================================================
    # Signal 1: Neural Prior
    draw_card(
        x=0.205, y=0.675, w=0.235, h=0.285,
        theme=colors['s1'],
        title='Signal 1: Domain Neural Prior',
        items=[
            'InLegalBERT Pre-trained Legal Transformer',
            r'Multi-Field Prompt Serialization ($[\mathrm{Q}]\oplus[\mathrm{E}]\oplus[\mathrm{A}]\oplus[\mathrm{N}]$)',
            r'Head-Tail Token Truncation ($160 + 350 \leq 512$)',
            'Two-Stage Progressive Layer Freezing',
            r'**Yields: Calibrated Prior $\mathbf{p}_{S1}$**'
        ],
        item_size=7.6, line_spacing=0.045
    )

    # Signal 2: LLM Coverage Auditor (NEW METHODOLOGY!)
    draw_card(
        x=0.205, y=0.355, w=0.235, h=0.290,
        theme=colors['s2'],
        title='Signal 2: LLM Judicial Auditor',
        items=[
            'Autonomous LLM-as-a-Judge Compliance Engine',
            r'Discrete Demand Itemization $\mathcal{Q} = \{Q_1, \dots, Q_N\}$',
            r'Dynamic Materiality Weights ($\omega_{\mathrm{core}} \approx 0.65, \omega_{\mathrm{aux}} \approx 0.35$)',
            'Compliance: Furnished, Partial, Exempt, Evaded',
            r'**Factual Coverage Ratio $\rho \in [0, 1] \to \mathbf{p}_{S2}$**',
            'Synthesizes Grounded Judicial Reasoning Traces'
        ],
        item_size=7.5, line_spacing=0.038
    )

    # Signal 3: Statutory Scanner
    draw_card(
        x=0.205, y=0.040, w=0.235, h=0.285,
        theme=colors['s3'],
        title='Signal 3: Statutory Scanner',
        items=[
            'Declarative RTI Legal Knowledge Base (Act of 2005)',
            'Section 8(1) Exemption Clauses (a)–(j)',
            'Section 24 Schedule II Immunity (26 Agencies)',
            'Section 20 Penal Defaults & Sec. 10 Severability',
            r'**Yields: Statutory Prior $\mathbf{p}_{S3}$**'
        ],
        item_size=7.6, line_spacing=0.045
    )

    # =========================================================================
    # 3. Weighted Probabilistic Consensus
    # =========================================================================
    draw_card(
        x=0.485, y=0.345, w=0.140, h=0.310,
        theme=colors['fused'],
        title='Weighted Consensus',
        items=[
            'Continuous Linear Fusion:',
            r'$\mathbf{p}_{\mathrm{fused}} = 0.50\,\mathbf{p}_{S1}$',
            r'$+ 0.30\,\mathbf{p}_{S2} + 0.20\,\mathbf{p}_{S3}$',
            '---',
            'Top Candidate:',
            r'**$\hat{y}_{\mathrm{fused}} = \arg\max_c p_{\mathrm{fused}}(c)$**'
        ],
        item_size=7.8, line_spacing=0.042
    )

    # =========================================================================
    # 4. 5-Tier Statutory Precedence Guardrails
    # =========================================================================
    draw_card(
        x=0.665, y=0.105, w=0.185, h=0.790,
        theme=colors['guard'],
        title='5-Tier Legal Guardrails',
        items=[
            '**Tier 1: Sec. 24 Immunity Veto**',
            '(Enforces DISMISSAL on Sec. 24 agencies)',
            '',
            '**Tier 2: Sec. 20 Penal Sanction**',
            '(Mandates PENALTY on malafide default)',
            '',
            '**Tier 3: Sec. 10 Severability Order**',
            r'(Awards PARTIAL RELIEF for $0.2 \leq \rho \leq 0.8$)',
            '',
            '**Tier 4: Evasion Overrule**',
            '(Forces DISCLOSURE if missing & no Sec. 8)',
            '',
            '**Tier 5: Sec. 8 Exemption Sustained**',
            '(Dismisses if legitimate exemption proven)',
            '---',
            '**Level 6: Standard Judicial Discretion**',
            r'(Confirms $\hat{y}_{\mathrm{fused}}$ if no boundary breached)'
        ],
        item_size=7.4, line_spacing=0.038
    )

    # =========================================================================
    # 5. Advisory Output
    # =========================================================================
    draw_card(
        x=0.875, y=0.22, w=0.115, h=0.56,
        theme=colors['output'],
        title='Advisory Output',
        items=[
            r'Advisory Verdict $\hat{y}^*$',
            '',
            'Confidence Metric',
            '',
            '**Override Flag (3.54%)**',
            '',
            'Cited RTI Section',
            '',
            'Draft Judicial Order',
            '',
            'Factual Audit Trail'
        ],
        item_size=7.8, line_spacing=0.045
    )

    # =========================================================================
    # ARROWS & LABELS
    # =========================================================================
    arrow_props = dict(arrowstyle="-|>", color=colors['arrow'], lw=2.0, mutation_scale=16)
    
    # Input D -> Signals 1, 2, 3
    x_in = 0.165
    y_in = 0.500
    x_sig = 0.205
    
    # Line to S1
    ax.annotate('', xy=(x_sig, 0.817), xytext=(x_in, y_in), arrowprops=arrow_props, zorder=1)
    # Line to S2
    ax.annotate('', xy=(x_sig, 0.500), xytext=(x_in, y_in), arrowprops=arrow_props, zorder=1)
    # Line to S3
    ax.annotate('', xy=(x_sig, 0.182), xytext=(x_in, y_in), arrowprops=arrow_props, zorder=1)

    # Signals -> Weighted Consensus
    x_s_out = 0.440
    x_f_in = 0.485
    y_f_in = 0.500

    lbl_box = dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E2E8F0', lw=0.8, alpha=0.95)

    # S1 -> Consensus
    ax.annotate('', xy=(x_f_in, 0.575), xytext=(x_s_out, 0.817), arrowprops=arrow_props, zorder=1)
    ax.text(0.460, 0.725, r'$w_1 = 0.50$', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=colors['s1']['title'], family=font_main,
            bbox=lbl_box, zorder=4)

    # S2 -> Consensus
    ax.annotate('', xy=(x_f_in, y_f_in), xytext=(x_s_out, 0.500), arrowprops=arrow_props, zorder=1)
    ax.text(0.4625, 0.522, r'$w_2 = 0.30$', ha='center', va='bottom',
            fontsize=8.5, fontweight='bold', color=colors['s2']['title'], family=font_main,
            bbox=lbl_box, zorder=4)

    # S3 -> Consensus
    ax.annotate('', xy=(x_f_in, 0.425), xytext=(x_s_out, 0.182), arrowprops=arrow_props, zorder=1)
    ax.text(0.460, 0.285, r'$w_3 = 0.20$', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color=colors['s3']['title'], family=font_main,
            bbox=lbl_box, zorder=4)

    # Consensus -> Guardrails
    ax.annotate('', xy=(0.665, 0.500), xytext=(0.625, 0.500), arrowprops=arrow_props, zorder=1)
    ax.text(0.645, 0.528, 'Candidate', ha='center', va='bottom',
            fontsize=8.0, fontweight='bold', color=colors['arrow'], family=font_main,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E2E8F0', lw=0.8, alpha=0.95), zorder=4)

    # Guardrails -> Output
    ax.annotate('', xy=(0.875, 0.500), xytext=(0.850, 0.500), arrowprops=arrow_props, zorder=1)
    ax.text(0.8625, 0.528, 'Enforced', ha='center', va='bottom',
            fontsize=8.0, fontweight='bold', color=colors['arrow'], family=font_main,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#E2E8F0', lw=0.8, alpha=0.95), zorder=4)

    # Signal 3 -> Guardrails (Statutory State Flags - Dashed Line)
    flag_line_props = dict(color=colors['guard_arrow'], linestyle='--', lw=1.8)
    ax.plot([0.3225, 0.3225], [0.040, 0.016], **flag_line_props, zorder=1)
    ax.plot([0.3225, 0.7575], [0.016, 0.016], **flag_line_props, zorder=1)
    ax.annotate('', xy=(0.7575, 0.105), xytext=(0.7575, 0.016),
                arrowprops=dict(arrowstyle="-|>", color=colors['guard_arrow'], linestyle='--', lw=1.8, mutation_scale=16), zorder=1)
    
    # Label on dashed line with crisp badge
    flag_lbl_box = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=colors['guard_arrow'], lw=1.0, alpha=0.95)
    ax.text(0.540, 0.016, r'Statutory State Flags $\mathcal{F}_{S3}$: $\langle \mathcal{E}_8, s_{24}, s_{20}, s_{10}, s_{\mathrm{ovr}} \rangle$',
            ha='center', va='center', fontsize=8.6, fontweight='bold', color=colors['guard_arrow'], family=font_main,
            bbox=flag_lbl_box, zorder=4)

    plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print(f"[SUCCESS] High-resolution architecture diagram written to: {output_path}")

if __name__ == '__main__':
    out_file = 'research_paper/figures/fig_architecture.png'
    create_architecture_diagram(out_file)

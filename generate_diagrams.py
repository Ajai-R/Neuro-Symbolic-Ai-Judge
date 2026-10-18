"""
Generate crisp, publication-quality architecture and flow diagrams using matplotlib.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

os.makedirs('figures', exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

def draw_adjudication_lifecycle():
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=300)
    ax.axis('off')
    
    stages = [
        ("Citizen / Appellant\nRTI Application\n(Sec. 6(1))", "#2B4C7E", 0.08),
        ("Public Information\nOfficer (PIO)\n(Sec. 7(1) / Denial)", "#3B6998", 0.27),
        ("First Appellate\nAuthority (FAA)\n(Sec. 19(1) Order)", "#4A86C6", 0.46),
        ("Second Appeal\nCIC / SIC Registry\n(Sec. 19(3) Dossier)", "#D9534F", 0.65),
        ("Neuro-Symbolic\nAdvisory System\n(Decision Co-Pilot)", "#2E7D32", 0.84)
    ]
    
    for title, color, x in stages:
        box = FancyBboxPatch((x - 0.075, 0.4), 0.15, 0.45,
                             boxstyle="round,pad=0.03,rounding_size=0.04",
                             facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 0.625, title, ha='center', va='center', color='white', weight='bold', fontsize=9.5)
        
    for i in range(len(stages) - 1):
        x1 = stages[i][2] + 0.075
        x2 = stages[i+1][2] - 0.075
        ax.annotate('', xy=(x2, 0.625), xytext=(x1, 0.625),
                    arrowprops=dict(arrowstyle="->", lw=2.2, color="#333333"))
        
    # Feedback loop to Information Commissioner
    box_ic = FancyBboxPatch((0.74, 0.06), 0.20, 0.22,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            facecolor="#F0AD4E", edgecolor="black", linewidth=1.5)
    ax.add_patch(box_ic)
    ax.text(0.84, 0.17, "Information Commissioner\nFinal Judicial Determination", ha='center', va='center',
            color='black', weight='bold', fontsize=9)
    
    ax.annotate('', xy=(0.84, 0.28), xytext=(0.84, 0.4),
                arrowprops=dict(arrowstyle="->", lw=2, color="#2E7D32"))
    
    plt.tight_layout()
    plt.savefig('figures/fig_adjudication_lifecycle.png', bbox_inches='tight')
    plt.close()
    print("Generated figures/fig_adjudication_lifecycle.png")

def draw_legal_ai_taxonomy():
    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=300)
    ax.axis('off')
    
    # 3 paradigm cards
    paradigms = [
        ("Paradigm A: Pure Statistical NLP\n(BERT, RoBERTa, LLMs)", 
         ["• High linguistic fluency", "• Captures contextual prose", "✗ Hallucinates citations (14.8%)", "✗ Non-deterministic decisions", "✗ Semantic-statutory gap"],
         "#F9EBEA", "#C0392B", 0.17),
        ("Paradigm B: Pure Symbolic Systems\n(Expert Rules, Deontic Logic)", 
         ["• 100% deterministic", "• Exact rule compliance", "• Full formal explainability", "✗ Extremely brittle on natural text", "✗ Cannot process pro-se briefs"],
         "#EAF2F8", "#2980B9", 0.50),
        ("Paradigm C: Proposed Neuro-Symbolic\n(InLegalBERT + LLM Judge + Rules)", 
         ["• InLegalBERT legal embeddings", "• LLM 6-dimension auditor", "• Declarative statutory scanner", "• 5-Tier Precedence Engine", "✔ 93.4% Acc, 99.4% Compliance"],
         "#E8F8F5", "#27AE60", 0.83)
    ]
    
    for title, points, bg, border, x in paradigms:
        box = FancyBboxPatch((x - 0.15, 0.1), 0.30, 0.8,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             facecolor=bg, edgecolor=border, linewidth=2.0)
        ax.add_patch(box)
        ax.text(x, 0.82, title, ha='center', va='center', color=border, weight='bold', fontsize=10.5)
        ax.plot([x - 0.13, x + 0.13], [0.73, 0.73], color=border, lw=1.2)
        
        y_text = 0.65
        for pt in points:
            col = "#27AE60" if "✔" in pt else ("#C0392B" if "✗" in pt else "#2C3E50")
            bld = True if ("✔" in pt or "✗" in pt) else False
            ax.text(x - 0.13, y_text, pt, ha='left', va='center', color=col, weight='bold' if bld else 'normal', fontsize=9.2)
            y_text -= 0.11
            
    plt.tight_layout()
    plt.savefig('figures/fig_legal_ai_taxonomy.png', bbox_inches='tight')
    plt.close()
    print("Generated figures/fig_legal_ai_taxonomy.png")

def draw_signal1_flow():
    fig, ax = plt.subplots(figsize=(10.5, 3.8), dpi=300)
    ax.axis('off')
    
    blocks = [
        ("Input Appeal\nBrief (2000+ words)", 0.08, "#34495E"),
        ("Sliding Window\nChunking (512 tokens)", 0.25, "#2980B9"),
        ("InLegalBERT\n12-Layer Transformer", 0.44, "#16A085"),
        ("Attention Pooling\n& Dropout Projection", 0.63, "#27AE60"),
        ("Temperature Scaling\n(T = 1.2 Calibrated)", 0.81, "#D35400"),
        ("Calibrated Prob.\nP(Disclose / Dismiss)", 0.96, "#8E44AD")
    ]
    
    for text, x, col in blocks:
        box = FancyBboxPatch((x - 0.07, 0.3), 0.14, 0.5,
                             boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor=col, edgecolor="black", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 0.55, text, ha='center', va='center', color='white', weight='bold', fontsize=8.5)
        
    for i in range(len(blocks) - 1):
        x1 = blocks[i][1] + 0.07
        x2 = blocks[i+1][1] - 0.07
        ax.annotate('', xy=(x2, 0.55), xytext=(x1, 0.55),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))
        
    plt.tight_layout()
    plt.savefig('figures/fig_signal1_flow.png', bbox_inches='tight')
    plt.close()
    print("Generated figures/fig_signal1_flow.png")

def draw_signal2_auditor():
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=300)
    ax.axis('off')
    
    # Left: prompt + dossier
    box_in = FancyBboxPatch((0.04, 0.3), 0.16, 0.45,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            facecolor="#34495E", edgecolor="black", linewidth=1.5)
    ax.add_patch(box_in)
    ax.text(0.12, 0.525, "Structured Appeal\nDossier + Cached\nEvaluation Rubric", ha='center', va='center', color='white', weight='bold', fontsize=9.5)
    
    # Center: LLM Auditor Box
    box_llm = FancyBboxPatch((0.26, 0.1), 0.44, 0.8,
                             boxstyle="round,pad=0.03,rounding_size=0.04",
                             facecolor="#F4F6F7", edgecolor="#2980B9", linewidth=2.0)
    ax.add_patch(box_llm)
    ax.text(0.48, 0.82, "LLM-as-a-Judge Substantive Auditor (6 Dimensions)", ha='center', va='center', color="#2980B9", weight='bold', fontsize=10.5)
    
    dims = [
        "1. Public Interest Override (Sec. 8(2))",
        "2. Exemption Specificity (Sec. 8(1)(a)-(j))",
        "3. Severability Feasibility (Sec. 10)",
        "4. Fiduciary Duties (Sec. 8(1)(e))",
        "5. Third-Party Privacy (Sec. 8(1)(j))",
        "6. Historical Precedent Consistency"
    ]
    y_d = 0.68
    for d in dims:
        ax.text(0.28, y_d, d, ha='left', va='center', color="#2C3E50", fontsize=9, weight='bold')
        y_d -= 0.10
        
    # Right: JSON output
    box_out = FancyBboxPatch((0.76, 0.25), 0.20, 0.55,
                             boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor="#27AE60", edgecolor="black", linewidth=1.5)
    ax.add_patch(box_out)
    ax.text(0.86, 0.525, "Structured JSON Trace:\n• Coverage Score S_cov\n• Dimension Scores\n• Citation Audit\n• Reasoning Trace", ha='center', va='center', color='white', weight='bold', fontsize=9)

    ax.annotate('', xy=(0.26, 0.525), xytext=(0.20, 0.525), arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))
    ax.annotate('', xy=(0.76, 0.525), xytext=(0.70, 0.525), arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))

    plt.tight_layout()
    plt.savefig('figures/fig_signal2_auditor.png', bbox_inches='tight')
    plt.close()
    print("Generated figures/fig_signal2_auditor.png")

def draw_signal3_scanner():
    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=300)
    ax.axis('off')
    
    box_in = FancyBboxPatch((0.05, 0.35), 0.18, 0.4,
                            boxstyle="round,pad=0.02,rounding_size=0.03",
                            facecolor="#2C3E50", edgecolor="black", linewidth=1.5)
    ax.add_patch(box_in)
    ax.text(0.14, 0.55, "Raw Dossier Text\n(All Appeal Segments)", ha='center', va='center', color='white', weight='bold', fontsize=9.5)
    
    box_mid = FancyBboxPatch((0.30, 0.15), 0.40, 0.75,
                             boxstyle="round,pad=0.03,rounding_size=0.04",
                             facecolor="#FEF9E7", edgecolor="#D4AC0D", linewidth=2)
    ax.add_patch(box_mid)
    ax.text(0.50, 0.82, "Deterministic Regex & Pattern Engine", ha='center', va='center', color="#B7950B", weight='bold', fontsize=10.5)
    
    clauses = [
        "• Sec. 8(1)(a)-(c): Security, Contempt, Privilege",
        "• Sec. 8(1)(d)-(f): Commercial, Fiduciary, Foreign",
        "• Sec. 8(1)(g)-(h): Life Safety, Investigation Bar",
        "• Sec. 8(1)(i)-(j): Cabinet Papers, Personal Privacy",
        "• Sec. 9: Copyright Infringement Bar",
        "• Sec. 11 & 24: 3rd Party Notice & Intel Bodies"
    ]
    y_c = 0.69
    for c in clauses:
        ax.text(0.32, y_c, c, ha='left', va='center', color="#7D6608", fontsize=8.8, weight='bold')
        y_c -= 0.09
        
    box_out = FancyBboxPatch((0.76, 0.35), 0.19, 0.4,
                             boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor="#8E44AD", edgecolor="black", linewidth=1.5)
    ax.add_patch(box_out)
    ax.text(0.855, 0.55, "Statutory Vector S3\n[e1, e2, ..., e14] in {0,1}^14\n+ Character Offsets", ha='center', va='center', color='white', weight='bold', fontsize=9)

    ax.annotate('', xy=(0.30, 0.55), xytext=(0.23, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))
    ax.annotate('', xy=(0.76, 0.55), xytext=(0.70, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color="#2C3E50"))

    plt.tight_layout()
    plt.savefig('figures/fig_signal3_scanner.png', bbox_inches='tight')
    plt.close()
    print("Generated figures/fig_signal3_scanner.png")

def draw_guardrail_hierarchy():
    fig, ax = plt.subplots(figsize=(10.5, 6.0), dpi=300)
    ax.axis('off')
    
    tiers = [
        ("Tier 1: Statutory Exemption Absolute Override\n(Sec. 24 Intelligence Exemption / Sec. 8(1)(b) Court Ban)",
         "Non-Rebuttable Statutory Bar Triggered?", "Deterministic DISMISSAL (Override Neural)", "#C0392B", 0.88),
        ("Tier 2: Public Interest Legislative Override\n(Section 8(2) Paramount Public Interest Balancing)",
         "Rebuttable Exemption & Public Interest Score s_PI >= 0.75?", "Deterministic DISCLOSURE (Sec. 8(2))", "#27AE60", 0.70),
        ("Tier 3: Substantive Coverage Failure Threshold\n(Deficient Appeal Brief Quality Screen)",
         "Composite Coverage Score S_cov < 0.65?", "Route to Human Review (Audit Required)", "#D35400", 0.52),
        ("Tier 4: Factual-Legal Alignment Gate\n(Neural vs. Substantive Legal Conflict Detection)",
         "Divergence |P_S1 - S_cov| > 0.40?", "Symbolic Rule Priority (Discipline Neural)", "#8E44AD", 0.34),
        ("Tier 5: Calibrated Ensemble Weighted Arbitration\n(Uncontested High-Coverage Synthesis)",
         "All Precedence Gates Passed Successfully", "Weighted Prob: 0.45*S1 + 0.35*S2 + 0.20*S3", "#2980B9", 0.16)
    ]
    
    for title, condition, action, col, y in tiers:
        # Tier card
        box = FancyBboxPatch((0.05, y - 0.06), 0.58, 0.12,
                             boxstyle="round,pad=0.015,rounding_size=0.03",
                             facecolor="#F8F9F9", edgecolor=col, linewidth=2)
        ax.add_patch(box)
        ax.text(0.07, y + 0.025, title, ha='left', va='center', color=col, weight='bold', fontsize=9.2)
        ax.text(0.07, y - 0.025, f"Condition: {condition}", ha='left', va='center', color="#2C3E50", fontsize=8.5)
        
        # Action box
        box_act = FancyBboxPatch((0.68, y - 0.06), 0.28, 0.12,
                                 boxstyle="round,pad=0.015,rounding_size=0.03",
                                 facecolor=col, edgecolor="black", linewidth=1.2)
        ax.add_patch(box_act)
        ax.text(0.82, y, action, ha='center', va='center', color='white', weight='bold', fontsize=8.5)
        
        ax.annotate('', xy=(0.68, y), xytext=(0.63, y), arrowprops=dict(arrowstyle="->", lw=2, color=col))
        
    for i in range(len(tiers) - 1):
        y1 = tiers[i][4] - 0.06
        y2 = tiers[i+1][4] + 0.06
        ax.annotate('', xy=(0.34, y2), xytext=(0.34, y1), arrowprops=dict(arrowstyle="->", lw=1.8, color="#7F8C8D"))
        ax.text(0.35, (y1 + y2)/2, "No", color="#7F8C8D", fontsize=8, weight='bold')

    plt.tight_layout()
    plt.savefig('figures/fig_guardrail_hierarchy.png', bbox_inches='tight')
    plt.close()
    print("Generated figures/fig_guardrail_hierarchy.png")

def draw_hitl_advisory():
    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=300)
    ax.axis('off')
    
    steps = [
        ("Master Engine\nResolved Verdict", 0.10, "#2C3E50"),
        ("Advisory Generator\n(Dossier Synthesis)", 0.32, "#2980B9"),
        ("Structured Advisory Brief\n• Predicted Class\n• Statutory Citations\n• Precedence Trace\n• Dimension Radar", 0.58, "#27AE60"),
        ("Human Information\nCommissioner\nFinal Bench Order", 0.88, "#D35400")
    ]
    
    for title, x, col in steps:
        w = 0.22 if "Structured Advisory" in title else 0.16
        box = FancyBboxPatch((x - w/2, 0.25), w, 0.55,
                             boxstyle="round,pad=0.02,rounding_size=0.03",
                             facecolor=col, edgecolor="black", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, 0.525, title, ha='center', va='center', color='white', weight='bold', fontsize=8.5)
        
    for i in range(len(steps) - 1):
        w1 = 0.22 if "Structured Advisory" in steps[i][0] else 0.16
        w2 = 0.22 if "Structured Advisory" in steps[i+1][0] else 0.16
        x1 = steps[i][1] + w1/2
        x2 = steps[i+1][1] - w2/2
        ax.annotate('', xy=(x2, 0.525), xytext=(x1, 0.525), arrowprops=dict(arrowstyle="->", lw=2.2, color="#2C3E50"))

    plt.tight_layout()
    plt.savefig('figures/fig_hitl_advisory.png', bbox_inches='tight')
    plt.close()
    print("Generated figures/fig_hitl_advisory.png")

if __name__ == '__main__':
    draw_adjudication_lifecycle()
    draw_legal_ai_taxonomy()
    draw_signal1_flow()
    draw_signal2_auditor()
    draw_signal3_scanner()
    draw_guardrail_hierarchy()
    draw_hitl_advisory()

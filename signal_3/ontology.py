"""
Signal 3: Declarative Statutory Knowledge Base & RTI Legal Ontology
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
Grounding: Right to Information Act, 2005 (Act No. 22 of 2005)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class RTISectionDefinition:
    section_id: str
    title: str
    statutory_clause: str
    legal_test: str
    primary_outcome_tendency: str
    regex_patterns: List[str]
    keywords: List[str]


class RTIStatutoryOntology:
    """
    Formal Declarative Legal Knowledge Base of the RTI Act, 2005.
    Captures statutory grounds for exemptions, penalties, severability, and jurisdictional limits.
    """

    # 1. Section 8(1) Statutory Exemptions
    SECTIONS_8_1: Dict[str, RTISectionDefinition] = {
        "8(1)(a)": RTISectionDefinition(
            section_id="8(1)(a)",
            title="Sovereignty, National Security & Strategic Interests",
            statutory_clause="Information, disclosure of which would prejudicially affect sovereignty and integrity of India, security, strategic, scientific or economic interests of the State...",
            legal_test="Does the requested disclosure compromise national security, defense capability, or international relations?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*a\s*\)',
                r'\bsection\s+8\s*\(1\)\s*\(a\)',
                r'\bsec(?:tion)?\s*\.?\s*8\s*\(?1\)?\s*\(?a\)?',
                r'\bsovereignty\s+and\s+integrity\b',
                r'\bsecurity\s+of\s+the\s+state\b',
                r'\bstrategic\s+interests?\b'
            ],
            keywords=["sovereignty", "national security", "strategic", "defense", "scientific interest"]
        ),
        "8(1)(b)": RTISectionDefinition(
            section_id="8(1)(b)",
            title="Contempt of Court & Judicial Prohibition",
            statutory_clause="Information which has been expressly forbidden to be published by any court of law or tribunal or the disclosure of which may constitute contempt of court...",
            legal_test="Has a competent court or judicial tribunal expressly enjoined or stayed publication?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*b\s*\)',
                r'\bsection\s+8\s*\(1\)\s*\(b\)',
                r'\bforbidden\s+by\s+(?:any\s+)?court\b',
                r'\bcontempt\s+of\s+court\b',
                r'\bsub-?\s*judice\b'
            ],
            keywords=["court of law", "tribunal", "contempt", "forbidden", "stay order", "subjudice"]
        ),
        "8(1)(c)": RTISectionDefinition(
            section_id="8(1)(c)",
            title="Breach of Parliamentary or Legislative Privilege",
            statutory_clause="Information, the disclosure of which would cause a breach of privilege of Parliament or the State Legislature...",
            legal_test="Does disclosure infringe legislative immunity or parliamentary privilege?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*c\s*\)',
                r'\bbreach\s+of\s+privilege\b',
                r'\bparliamentary\s+privilege\b',
                r'\bstate\s+legislature\s+privilege\b'
            ],
            keywords=["breach of privilege", "parliament", "legislature", "privilege"]
        ),
        "8(1)(d)": RTISectionDefinition(
            section_id="8(1)(d)",
            title="Commercial Confidence, Trade Secrets & Intellectual Property",
            statutory_clause="Information including commercial confidence, trade secrets or intellectual property, the disclosure of which would harm the competitive position of a third party...",
            legal_test="Would disclosure cause unfair competitive harm to commercial bidding, intellectual property, or trade secrets without demonstrable public interest?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*d\s*\)',
                r'\bsection\s+8\s*\(1\)\s*\(d\)',
                r'\bcommercial\s+confidence\b',
                r'\btrade\s+secrets?\b',
                r'\bintellectual\s+property\b',
                r'\bcompetitive\s+position\b'
            ],
            keywords=["commercial confidence", "trade secret", "intellectual property", "competitive position", "tender pricing"]
        ),
        "8(1)(e)": RTISectionDefinition(
            section_id="8(1)(e)",
            title="Fiduciary Relationship",
            statutory_clause="Information available to a person in his fiduciary relationship, unless the competent authority is satisfied that the larger public interest warrants the disclosure...",
            legal_test="Is the record held in a relationship of trust and confidentiality (e.g., doctor-patient, lawyer-client, bank-customer)?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*e\s*\)',
                r'\bsection\s+8\s*\(1\)\s*\(e\)',
                r'\bfiduciary\s+relationship\b',
                r'\bfiduciary\s+capacity\b',
                r'\bin\s+trust\b'
            ],
            keywords=["fiduciary", "trust", "banker customer", "client confidentiality"]
        ),
        "8(1)(f)": RTISectionDefinition(
            section_id="8(1)(f)",
            title="Foreign Government Confidence",
            statutory_clause="Information received in confidence from foreign Government...",
            legal_test="Was the information transmitted under confidential diplomatic treaty or bilateral pact?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*f\s*\)',
                r'\bforeign\s+government\b',
                r'\bin\s+confidence\s+from\s+foreign\b'
            ],
            keywords=["foreign government", "diplomatic", "treaty"]
        ),
        "8(1)(g)": RTISectionDefinition(
            section_id="8(1)(g)",
            title="Endangerment of Life or Physical Safety & Law Enforcement Sources",
            statutory_clause="Information, the disclosure of which would endanger the life or physical safety of any person or identify the source of information or assistance given in confidence for law enforcement...",
            legal_test="Does disclosure expose informants, witnesses, examiners, or law officers to physical threat?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*g\s*\)',
                r'\bphysical\s+safety\b',
                r'\bendanger\s+(?:the\s+)?life\b',
                r'\bsource\s+of\s+information\b'
            ],
            keywords=["endanger life", "physical safety", "informer", "witness protection", "source of assistance"]
        ),
        "8(1)(h)": RTISectionDefinition(
            section_id="8(1)(h)",
            title="Impeding Investigation or Prosecution of Offenders",
            statutory_clause="Information which would impede the process of investigation or apprehension or prosecution of offenders...",
            legal_test="Is there an active, ongoing police or departmental inquiry that would be derailed by disclosure?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*h\s*\)',
                r'\bsection\s+8\s*\(1\)\s*\(h\)',
                r'\bimpede\s+(?:the\s+)?process\s+of\s+investigation\b',
                r'\bongoing\s+investigation\b',
                r'\bprosecution\s+of\s+offenders\b',
                r'\bpending\s+inquiry\b'
            ],
            keywords=["investigation", "impede", "apprehension", "prosecution", "charge sheet", "ongoing inquiry"]
        ),
        "8(1)(i)": RTISectionDefinition(
            section_id="8(1)(i)",
            title="Cabinet Papers & Deliberations of Council of Ministers",
            statutory_clause="Cabinet papers including records of deliberations of the Council of Ministers, Secretaries and other officers...",
            legal_test="Are these pre-decisional Cabinet records before decisions have been taken and implemented?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*i\s*\)',
                r'\bcabinet\s+papers?\b',
                r'\bcouncil\s+of\s+ministers\b',
                r'\bdeliberations\s+of\s+the\s+council\b'
            ],
            keywords=["cabinet papers", "council of ministers", "secretaries deliberation"]
        ),
        "8(1)(j)": RTISectionDefinition(
            section_id="8(1)(j)",
            title="Personal Information & Unwarranted Invasion of Privacy",
            statutory_clause="Information which relates to personal information the disclosure of which has no relationship to any public activity or interest, or which would cause unwarranted invasion of the privacy of the individual...",
            legal_test="Does the record constitute third-party personal details (medical records, ACRs, bank accounts, personal phone numbers) lacking overriding public interest?",
            primary_outcome_tendency="APPEAL_DISMISSED",
            regex_patterns=[
                r'\b8\s*\(\s*1\s*\)\s*\(\s*j\s*\)',
                r'\bsection\s+8\s*\(1\)\s*\(j\)',
                r'\bpersonal\s+information\b',
                r'\binvasion\s+of\s+privacy\b',
                r'\bthird\s+party\s+privacy\b',
                r'\bunwarranted\s+invasion\b',
                r'\bpersonal\s+data\b'
            ],
            keywords=["personal information", "privacy", "third party", "unwarranted invasion", "medical records", "salary slip", "acr"]
        )
    }

    # 2. Section 24 Second Schedule (Exempt Intelligence & Security Organizations)
    # Complete canonical list under RTI Act, 2005 (Central agencies)
    SEC24_EXEMPT_AGENCIES: Set[str] = {
        "intelligence bureau", "ib", "cabinet secretariat", "research and analysis wing", "r&aw",
        "directorate of revenue intelligence", "dri", "central economic intelligence bureau", "ceib",
        "directorate of enforcement", "enforcement directorate",
        "narcotics control bureau", "ncb", "aviation research centre", "arc",
        "special frontier force", "sff", "border security force", "bsf",
        "central reserve police force", "crpf", "indo-tibetan border police", "itbp",
        "central industrial security force", "cisf", "national security guard", "nsg",
        "assam rifles", "special service bureau", "ssb", "national security council secretariat", "nscs",
        "financial intelligence unit", "fiu-ind", "fiu", "defence research and development organisation", "drdo",
        "border road development board", "brdb", "national technical research organisation", "ntro",
        "central bureau of investigation", "cbi", "national investigation agency", "nia",
        "strategic forces command"
    }

    # 3. Section 20 Statutory Penalties (Triggers for PENALTY_IMPOSED)
    # Mandates Rs. 250/day up to Rs. 25,000 for unreasonable refusal, delay > 30 days, or destroyed records
    SEC20_PENALTY_INDICATORS: List[str] = [
        "section 20", "sec 20", "penalty under section 20", "show cause notice",
        "malafide denial", "malafidely refused", "knowingly gave incorrect",
        "deliberately obstructed", "destroyed information", "persistent non-compliance",
        "penalized under section 20(1)", "recommend disciplinary action under 20(2)",
        "unreasonable delay", "explained delay", "compensation under 19(8)(b)"
    ]

    # 4. Section 10 Severability Principle (Triggers for PARTIAL_RELIEF)
    # Mandates disclosure of non-exempt portions when exempt parts can reasonably be severed
    SEC10_SEVERABILITY_INDICATORS: List[str] = [
        "section 10", "severability", "severed", "partially exempt",
        "redacted copy", "masked copy", "severable portions",
        "part of the information", "redaction", "blacked out",
        "severing the confidential"
    ]

    # 5. Section 8(2) Public Interest Override Keywords
    PUBLIC_INTEREST_OVERRIDE: List[str] = [
        "larger public interest", "public interest warrants",
        "overrides the exemption", "public interest outweighs", "section 8(2)"
    ]

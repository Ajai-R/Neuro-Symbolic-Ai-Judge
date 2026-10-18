"""
Signal 3: Statutory Fact & Boundary Scanner
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set

from .ontology import RTIStatutoryOntology, RTISectionDefinition


@dataclass
class StatutoryFacts:
    """Structured legal facts extracted from the RTI dispute record."""
    case_id: str
    exemptions_detected: List[str] = field(default_factory=list)
    has_valid_sec8_claim: bool = False
    is_sec24_organization: bool = False
    sec24_agency_name: Optional[str] = None
    human_rights_or_corruption_alleged: bool = False
    sec20_penalty_triggered: bool = False
    sec20_indicators: List[str] = field(default_factory=list)
    sec10_severability_indicated: bool = False
    public_interest_override_claimed: bool = False
    statutory_summary: str = ""


class StatutoryScanner:
    """
    Parses unstructured case narratives, citizen queries, and authority metadata
    against the formal RTI Act, 2005 Ontology.
    """
    def __init__(self):
        self.ontology = RTIStatutoryOntology

    def scan_case(
        self,
        case_narrative: str,
        information_sought: Optional[str] = None,
        public_authority: Optional[str] = None,
        exemptions_cited: Optional[str] = None,
        case_id: Optional[str] = None
    ) -> StatutoryFacts:
        """
        Extracts formal statutory facts and legal boundaries from the case record.
        """
        narrative_lower = str(case_narrative or "").lower()
        query_lower = str(information_sought or "").lower()
        authority_lower = str(public_authority or "").lower()
        combined_text = f"{query_lower}\n{narrative_lower}"

        detected_exemptions: Set[str] = set()

        # 1. Check structured exemptions_cited field first
        if exemptions_cited:
            try:
                parsed = json.loads(exemptions_cited) if isinstance(exemptions_cited, str) else exemptions_cited
                if isinstance(parsed, list):
                    for ex in parsed:
                        ex_clean = str(ex).strip()
                        if ex_clean in self.ontology.SECTIONS_8_1:
                            detected_exemptions.add(ex_clean)
            except Exception:
                pass

        # 2. Regex and pattern scan across combined text for Section 8(1)
        for sec_id, sec_def in self.ontology.SECTIONS_8_1.items():
            for pattern in sec_def.regex_patterns:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    detected_exemptions.add(sec_id)
                    break

        sorted_exemptions = sorted(list(detected_exemptions))
        has_sec8 = len(sorted_exemptions) > 0

        # 3. Section 24 Jurisdictional Bar Check (Exempt Security Agencies)
        is_sec24 = False
        matched_agency = None
        has_sec24_statutory_plea = bool(re.search(r'\b(?:section\s*24|sec\s*24|second\s+schedule|exempt(?:ed)?\s+(?:organization|agency|from\s+the\s+purview))\b', combined_text))
        for agency in self.ontology.SEC24_EXEMPT_AGENCIES:
            # Check authority name directly
            if re.search(r'\b' + re.escape(agency) + r'\b', authority_lower):
                is_sec24 = True
                matched_agency = agency.upper()
                break
            # If in narrative, require explicit Section 24 / Second Schedule context
            elif has_sec24_statutory_plea and len(agency) > 3 and re.search(r'\b' + re.escape(agency) + r'\b', narrative_lower):
                is_sec24 = True
                matched_agency = agency.upper()
                break

        # Check proviso to Section 24: Allegations of corruption or human rights violations override Section 24
        hr_corruption = False
        if is_sec24:
            if re.search(r'\b(?:corruption|human\s+rights\s+violation|human\s+rights)\b', combined_text):
                hr_corruption = True

        # 4. Section 20 Statutory Penalty Check
        sec20_indicators = []
        for term in self.ontology.SEC20_PENALTY_INDICATORS:
            if term in narrative_lower:
                sec20_indicators.append(term)
        
        # Penalty triggered if explicit show-cause notice or malafide obstruction detected
        sec20_triggered = len(sec20_indicators) >= 2 or any(
            t in sec20_indicators for t in ["show cause notice", "malafide denial", "penalized under section 20(1)"]
        )

        # 5. Section 10 Severability Check
        sec10_indicated = any(term in combined_text for term in self.ontology.SEC10_SEVERABILITY_INDICATORS)

        # 6. Section 8(2) Public Interest Override Check
        pi_override = any(term in combined_text for term in self.ontology.PUBLIC_INTEREST_OVERRIDE)

        # 7. Compose Statutory Summary
        summary_parts = []
        if is_sec24:
            if hr_corruption:
                summary_parts.append(f"Section 24 Agency ({matched_agency}) detected, but corruption/human-rights proviso engaged.")
            else:
                summary_parts.append(f"Section 24 Jurisdictional Bar: Public Authority ({matched_agency}) is an exempt security agency.")
        if has_sec8:
            summary_parts.append(f"Section 8(1) Exemption(s) Cited: {', '.join(sorted_exemptions)}.")
        else:
            summary_parts.append("No Section 8(1) statutory exemption cited by PIO.")
        if sec20_triggered:
            summary_parts.append(f"Section 20 Penalty grounds detected ({', '.join(sec20_indicators[:2])}).")
        if sec10_indicated:
            summary_parts.append("Section 10 Severability grounds identified in arguments.")

        statutory_summary = " ".join(summary_parts)

        return StatutoryFacts(
            case_id=case_id or "ANON_CASE",
            exemptions_detected=sorted_exemptions,
            has_valid_sec8_claim=has_sec8,
            is_sec24_organization=is_sec24 and (not hr_corruption),
            sec24_agency_name=matched_agency,
            human_rights_or_corruption_alleged=hr_corruption,
            sec20_penalty_triggered=sec20_triggered,
            sec20_indicators=sec20_indicators,
            sec10_severability_indicated=sec10_indicated,
            public_interest_override_claimed=pi_override,
            statutory_summary=statutory_summary
        )

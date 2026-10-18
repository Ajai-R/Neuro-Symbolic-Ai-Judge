"""
Signal 2: Evidence & Question Item Extraction Module
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30
"""

import re
import string
from typing import List, Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class LegalSentenceSplitter:
    """
    Splits legal texts into candidate sentences while preserving statutory abbreviations
    (e.g., CPIO, PIO, Sec., Rs., No., Sh., vs., w.e.f., para., Hon'ble).
    """
    # Negative lookbehind pattern for common Indian administrative & legal abbreviations
    ABBR_PATTERN = re.compile(
        r'(?<!\bCPIO)(?<!\bPIO)(?<!\bFAA)(?<!\bCIC)(?<!\bSIC)(?<!\bSec)(?<!\bRs)(?<!\bNo)(?<!\bSh)(?<!\bSmt)'
        r'(?<!\bvs)(?<!\bpara)(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bProf)(?<!\bHon\'ble)(?<!\bw\.e\.f)(?<!\be\.g)(?<!\bi\.e)'
        r'(?<!\bArt)(?<!\bGovt)(?<!\bDept)(?<!\bLtd)(?<!\bPvt)'
        r'(?<=\.|\?|\!)\s+(?=[A-Z0-9"\'\(\[])',
        re.IGNORECASE
    )

    @classmethod
    def split(cls, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        
        # Normalize newlines
        clean_text = text.replace('\r\n', '\n').strip()
        paragraphs = [p.strip() for p in clean_text.split('\n') if p.strip()]
        
        sentences = []
        for p in paragraphs:
            # Check if paragraph is a single bullet point
            raw_sents = cls.ABBR_PATTERN.split(p)
            for s in raw_sents:
                s_clean = s.strip()
                if len(s_clean) > 8:  # filter negligible fragments
                    sentences.append(s_clean)
        return sentences


class QuestionItemizer:
    """
    Deconstructs citizen queries into distinct question elements (Q1, Q2, ...).
    Combines rule-based pattern matching (for numbered/bulleted lists) with
    syntactic clause segmentation and FLAN-T5 fallback.
    """
    # Match patterns like: "1.", "(1)", "1)", "(i)", "a)", "Point 1:", "Item 1:"
    ENUM_REGEX = re.compile(
        r'(?:^|\n|\s)(?:'
        r'\d{1,2}[\.\)]|\(\d{1,2}\)|'
        r'\([a-zA-Z]\)|[a-zA-Z]\)|\b[a-zA-Z]\.|'
        r'\([ivxIVX]{1,4}\)|[ivxIVX]{1,4}\.|\b(?:Point|Item)\s*\d+[:\.\-]?'
        r')\s+',
        re.MULTILINE
    )

    CONJUNCTIVE_DELIMITERS = [
        ";\n", "; ", 
        ", and also ", ", along with ", ", as well as ",
        " and copy of ", " and details of ", " and information on "
    ]

    @classmethod
    def extract_items_rule_based(cls, query_text: str) -> List[str]:
        """
        Fast, zero-latency deterministic extraction for structured queries.
        """
        if not query_text or not isinstance(query_text, str):
            return ["Information requested by citizen"]
        
        text = query_text.strip()
        
        # 1. Check for explicit enumeration
        matches = list(cls.ENUM_REGEX.finditer(text))
        if len(matches) >= 2:
            items = []
            for i in range(len(matches)):
                start = matches[i].end()
                end = matches[i + 1].start() if (i + 1 < len(matches)) else len(text)
                clause = text[start:end].strip().strip(';.,-')
                if clause and len(clause) > 5:
                    items.append(clause)
            if len(items) >= 2:
                return items

        # 2. Check for semicolon or newline splits
        if "\n" in text:
            lines = [l.strip().lstrip('*-•>0123456789.) ').strip(';.,') for l in text.split("\n")]
            valid_lines = [l for l in lines if len(l) > 10]
            if len(valid_lines) >= 2:
                return valid_lines

        if ";" in text:
            semi_parts = [p.strip().strip(';.,') for p in text.split(";")]
            valid_parts = [p for p in semi_parts if len(p) > 10]
            if len(valid_parts) >= 2:
                return valid_parts

        # 3. Check for multi-item conjunctive clauses
        for delim in cls.CONJUNCTIVE_DELIMITERS:
            if delim in text:
                parts = [p.strip().strip(';.,') for p in text.split(delim)]
                valid = [p for p in parts if len(p) > 12]
                if len(valid) >= 2:
                    return valid

        # Default: Single coherent request
        clean_single = text.lstrip('*-•>0123456789.) ').strip()
        return [clean_single] if clean_single else [query_text]


class FlanT5Extractor:
    """
    Neural Evidence & Item Extractor using google/flan-t5.
    Can be used for complex unstructured queries and generating plain-English summaries.
    """
    def __init__(self, model_name: str = "google/flan-t5-base", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def load_model(self):
        if not self._is_loaded:
            print(f"Loading FLAN-T5 ({self.model_name}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True

    def decompose_query(self, query: str) -> List[str]:
        """
        Uses FLAN-T5 to itemize free-form, unformatted citizen requests.
        """
        # First try rule-based extraction
        items = QuestionItemizer.extract_items_rule_based(query)
        if len(items) > 1:
            return items

        # If rule-based yielded only 1 item and text is long, use FLAN-T5 if available
        if len(query.split()) > 25:
            try:
                self.load_model()
                prompt = (
                    f"Task: Extract each distinct document, record, or piece of information requested by the citizen.\n"
                    f"Format: Return a numbered list (1., 2., 3.).\n"
                    f"RTI Query: {query}\n"
                    f"Items:"
                )
                inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=150, temperature=0.0)
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                t5_items = QuestionItemizer.extract_items_rule_based(decoded)
                if len(t5_items) >= 2:
                    return t5_items
            except Exception as e:
                # Graceful fallback to rule-based
                pass

        return items

    def summarize_pio_action(self, question: str, best_response: str) -> str:
        """
        Generates a concise legal summary explaining the PIO's action relative to the question.
        """
        if not best_response:
            return "No substantive response or record was provided by the PIO for this item."

        try:
            self.load_model()
            prompt = (
                f"Citizen asked for: {question}\n"
                f"PIO submitted: {best_response}\n"
                f"Question: Did the PIO provide, deny, or evade this request? Explain in one short sentence:"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=60, temperature=0.0)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception:
            return f"PIO submitted: '{best_response[:120]}...'"

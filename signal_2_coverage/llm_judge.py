"""
LLM-as-a-Judge for Signal 2: Substantive Information Disclosure & Compliance Auditor
Project: Explainable Neuro-Symbolic AI Advisory System for RTI Decision Outcome Prediction
Batch ID: 30

Replaces superficial cosine semantic similarity with an LLM Judicial Auditor that:
1. Decomposes citizen RTI queries into discrete, itemized question demands (Q_1 ... Q_N).
2. Assigns dynamic importance weightings:
   - CORE_OPERATIVE_DEMAND (weight ~ 0.65): Substantive records, audit reports, file notings, sanctions.
   - AUXILIARY_INQUIRY (weight ~ 0.35): Procedural dates, designations, inward diary numbers.
3. Evaluates whether the requested information was actually furnished vs. withheld/evaded:
   - FURNISHED (1.0): Certified copies, specific data, or substantive records were provided.
   - PARTIALLY_FURNISHED (0.5): Incomplete reply or summary without primary records.
   - WITHHELD_WITH_STATUTORY_CLAIM (0.0): Refusal claiming Section 8(1) or Section 24.
   - NOT_PROVIDED_EVADED (0.0): Administrative silence, deflection, or non-response.
4. Calculates a weighted factual compliance score in [0.0, 1.0] and produces a plain-English judicial audit trace.
"""

import os
import re
import json
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

try:
    import dotenv
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv.load_dotenv(os.path.join(_project_root, ".env"), override=False)
    dotenv.load_dotenv(override=False)
except Exception:
    pass


JUDICIAL_AUDITOR_SYSTEM_PROMPT = """You are a Senior Judicial Registrar and Legal Compliance Auditor for the Central Information Commission (CIC) under the Right to Information Act, 2005.

Your task is to conduct an authoritative question-by-question substantive compliance audit of an RTI dispute:
1. Examine the Citizen's Information Sought and itemize every question/demand asked (Q1, Q2, ...).
2. For each question asked, analyze the Public Authority's Response to determine what was replied and what was not replied:
   - "FURNISHED": The question was fully replied and requested records/data were supplied. (Compliance Score = 1.0)
   - "PARTIALLY_FURNISHED": The question was partially replied; some information was given but key portions omitted. (Compliance Score = 0.5)
   - "WITHHELD_WITH_STATUTORY_CLAIM": Not replied; expressly denied citing statutory exemptions (e.g., Section 8(1)(j), 8(1)(d), 8(1)(e), Section 24). (Compliance Score = 0.0)
   - "NOT_PROVIDED_EVADED": Not replied; evaded, claimed 'not available', omitted, delayed, or silent without statutory justification. (Compliance Score = 0.0)
3. Assign legal importance to each question asked:
   - "CORE_OPERATIVE_DEMAND": Substantive files, inspection reports, financial accounts, tender sanction files, file notings, certified lists.
   - "AUXILIARY_INQUIRY": Procedural queries, dates of receipt, dispatch diary numbers, designations of handling staff.
4. Extract the exact reply excerpt of what was replied (or "Not replied / No reply on record").
5. Provide a concise 1-sentence legal reason explaining what was replied or why it was not replied.

You MUST reply with ONLY a single valid JSON object with this exact schema:
{
  "total_items": <int>,
  "items": [
    {
      "query_index": <int>,
      "question_text": "<exact demand asked by citizen>",
      "importance": "CORE_OPERATIVE_DEMAND" | "AUXILIARY_INQUIRY",
      "importance_weight": <float between 0.3 and 0.7>,
      "status": "FURNISHED" | "PARTIALLY_FURNISHED" | "WITHHELD_WITH_STATUTORY_CLAIM" | "NOT_PROVIDED_EVADED",
      "compliance_score": <1.0 | 0.5 | 0.0>,
      "matched_reply_excerpt": "<exact excerpt of what was replied or 'Not replied'>",
      "legal_reason": "<concise explanation of what was replied or why it was not replied>"
    }
  ],
  "audit_summary": "<concise judicial evaluation summary explaining what was replied vs not replied>"
}
Do not include any introductory or concluding text outside the JSON object.
"""


class LLMJudicialAuditor:
    """
    Substantive Disclosure & Compliance Auditor powered by OpenRouter API
    (Default: NVIDIA Nemotron 3 Ultra 550B) with zero-downtime heuristic fallback.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.model_name = model_name or os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        self.api_key = (api_key or os.environ.get("OPENROUTER_API_KEY", "")).strip()
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.device = device or ("cuda" if self._check_cuda() else "cpu")
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_attempted = False
        self.last_active_model = "Judicial Compliance Auditor (LLM Judge)"
        self.active_openrouter_model = None
        self.active_reasoning_model = None

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _get_models_to_try(self) -> List[str]:
        """
        Returns prioritised list of OpenRouter models to reduce API-side downtime.
        Tries user's chosen primary model first, followed by top free models.
        """
        primary = self.model_name or os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
        candidates = [
            primary,
            "nvidia/nemotron-3-super-120b-a12b:free",
            "nvidia/nemotron-3.5-lightning:free",
            "liquid/lfm-2.5-2.6b:free"
        ]
        seen = set()
        ordered = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        return ordered

    def _get_api_keys_to_try(self) -> List[str]:
        """
        Returns list of configured OpenRouter API keys for automatic failover.
        If primary key hits rate limit (429), the auditor automatically switches to backup key.
        """
        keys = []
        if self.api_key and "..." not in self.api_key and len(self.api_key) > 15:
            keys.append(self.api_key)
        for var_name in ["OPENROUTER_API_KEY", "OPENROUTER_BACKUP_API_KEY"]:
            val = os.environ.get(var_name, "").strip()
            if val and "..." not in val and len(val) > 15 and val not in keys:
                keys.append(val)
        return keys

    def _infer_google_api(self, system_prompt: str, user_content: str, max_tokens: int = 800) -> Optional[str]:
        """
        Executes inference via Google Generative AI REST API (supports Gemini & Gemma models)
        using GEMINI_API_KEY as an automatic failover provider when OpenRouter quota is exceeded.
        """
        key = (self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        if not key or len(key) < 15:
            return None

        # Prioritize fast, high-quota models available on the Google Generative Language key
        models_to_try = [
            "gemini-flash-lite-latest",
            "gemma-4-26b-a4b-it",
            "gemini-flash-latest"
        ]

        prompt = (
            f"{system_prompt}\n\n"
            f"{user_content}\n\n"
            f"IMPORTANT: Output ONLY the final response. Do NOT output thinking steps, reasoning outlines, or markdown bullet labels."
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": max_tokens
            }
        }
        encoded_data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            try:
                req = urllib.request.Request(url, data=encoded_data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            ans = parts[0]["text"].strip()
                            if ans:
                                print(f"[Signal 2 Google AI] Successfully generated inference using {model_name}.")
                                return ans
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                print(f"[Signal 2 Google AI] {model_name} HTTP {e.code} ({err_body[:60]})")
            except Exception as e:
                print(f"[Signal 2 Google AI] {model_name} error: {e}")

        return None

    @staticmethod
    def _clean_case_finding(raw_finding: str, outcome_display: str) -> Optional[str]:
        """
        Cleans and normalizes generated Judicial Reasoning into 3 to 4 sentences
        strictly beginning with '<Outcome> was chosen because...'.
        """
        if not raw_finding:
            return None

        text = raw_finding.strip()
        if "Case Finding:" in text:
            text = text.split("Case Finding:")[-1].strip()

        # Find the statement starting with the outcome
        target_start = f"{outcome_display} was chosen because"
        pos = text.lower().rfind(target_start.lower())
        if pos != -1:
            text = text[pos:].strip()

        # Remove chain-of-thought headings, asterisks, bullet prefixes like *Sentence 1:*
        text = re.sub(r'(\*+|\b)Sentence\s*\d+[:\*]*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\*`#]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        elif text.startswith('"'):
            text = text[1:].strip()

        # Ensure placeholders were not parroted
        if "[what the PIO did" in text or "[Relevant legal provisions" in text:
            return None

        if len(text) > 25:
            return text
        return None

    def _infer_openrouter(self, query: str, narrative: str) -> Optional[Dict]:
        """
        Executes substantive compliance audit prompt via OpenRouter API with Google AI fallback.
        Attempts primary model and cascades across backup models, backup OpenRouter keys,
        and Google AI (Gemini/Gemma) if rate limits (429) occur.
        """
        api_keys = self._get_api_keys_to_try()
        user_content = (
            f"CITIZEN INFORMATION SOUGHT / QUERY ASKED:\n{query}\n\n"
            f"PUBLIC AUTHORITY RESPONSE / REPLY GIVEN BY RTI:\n{narrative}\n\n"
            f"Conduct the substantive compliance audit and output strictly the valid JSON object:"
        )

        if api_keys:
            url = "https://openrouter.ai/api/v1/chat/completions"

            for key_idx, key in enumerate(api_keys):
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5173",
                    "X-Title": "RTI Neuro-Symbolic Judicial Auditor"
                }
                key_mask = f"...{key[-6:]}" if len(key) > 6 else "key"

                for candidate_model in self._get_models_to_try():
                    payload = {
                        "model": candidate_model,
                        "messages": [
                            {"role": "system", "content": JUDICIAL_AUDITOR_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 1200
                    }

                    try:
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(payload).encode("utf-8"),
                            headers=headers,
                            method="POST"
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            data = json.loads(resp.read().decode("utf-8"))
                            choices = data.get("choices")
                            if choices and len(choices) > 0 and "message" in choices[0]:
                                msg = choices[0]["message"]
                                response_text = msg.get("content") or msg.get("reasoning") or ""
                                parsed = self._extract_json(response_text)
                                if parsed and "items" in parsed:
                                    print(f"[Signal 2 OpenRouter] Successfully audited case using model: {candidate_model} (key {key_mask})")
                                    self.active_openrouter_model = candidate_model
                                    return parsed
                            elif "error" in data:
                                err_msg = data["error"].get("message", "overload")
                                code = data["error"].get("code", 0)
                                print(f"[Signal 2 OpenRouter] {candidate_model} (key {key_mask}) error: {err_msg[:60]}")
                                if code == 429 or "rate limit" in err_msg.lower():
                                    print(f"[Signal 2 OpenRouter] Rate limit reached on key {key_mask}; failing over to next API key...")
                                    break
                    except urllib.error.HTTPError as e:
                        err_body = e.read().decode("utf-8", errors="ignore")
                        print(f"[Signal 2 OpenRouter] {candidate_model} (key {key_mask}) HTTP {e.code} ({err_body[:60]})")
                        if e.code == 429:
                            print(f"[Signal 2 OpenRouter] HTTP 429 on key {key_mask}; failing over to next API key...")
                            break
                    except Exception as e:
                        print(f"[Signal 2 OpenRouter] {candidate_model} request failed ({e}); cascading to next model...")

        # Fallback to Google Generative AI (Gemini / Gemma) if OpenRouter keys are exhausted
        google_json_text = self._infer_google_api(JUDICIAL_AUDITOR_SYSTEM_PROMPT, user_content, max_tokens=1200)
        if google_json_text:
            parsed = self._extract_json(google_json_text)
            if parsed and "items" in parsed:
                print("[Signal 2 Google AI] Substantive compliance audit completed successfully via Google AI.")
                self.active_openrouter_model = "Google Gemma / Gemini AI"
                return parsed

        return None

    def load_model(self) -> bool:
        """
        Loads local LLM into GPU memory if local model weights are present.
        """
        if self.is_loaded:
            return True
        if self._load_attempted:
            return False

        self._load_attempted = True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from huggingface_hub import try_to_load_from_cache

            # Check if model.safetensors is fully downloaded
            cached_weights = try_to_load_from_cache("Qwen/Qwen2.5-1.5B-Instruct", "model.safetensors")
            if not cached_weights or not isinstance(cached_weights, str) or not os.path.exists(cached_weights):
                self.is_loaded = False
                return False

            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                trust_remote_code=True,
                local_files_only=True
            )
            
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                local_files_only=True
            ).to(self.device)

            self.model.eval()
            self.is_loaded = True
            return True

        except Exception:
            self.is_loaded = False
            return False

    def audit_case(
        self,
        information_sought: str,
        case_narrative: str,
        case_id: Optional[str] = None
    ) -> Dict:
        """
        Conducts substantive disclosure audit on RTI query vs public authority response.
        Prioritizes OpenRouter Nemotron 550B, falling back to high-precision legal heuristics.
        """
        start_time = time.time()
        query = str(information_sought or "").strip()
        narrative = str(case_narrative or "").strip()

        if not query:
            query = "General information requested regarding public administrative records."
        if not narrative:
            narrative = "No substantive reply or records produced by the respondent public authority."

        # 1. Execute LLM Judicial Audit via API key (OpenRouter / Google Gemini)
        llm_result = None
        judge_display = "Judicial Compliance Auditor (LLM Judge)"

        current_key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        has_gemini = bool(self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "").strip())
        if (current_key and "..." not in current_key and len(current_key) > 15) or has_gemini:
            print(f"[Signal 2 LLM Judge] Auditing case with AI Judicial Judge...")
            llm_result = self._infer_openrouter(query, narrative)
            if llm_result:
                active_mod = self.active_openrouter_model or self.model_name
                if "gemma" in active_mod.lower() or "gemini" in active_mod.lower() or "google" in active_mod.lower():
                    judge_display = f"{active_mod}"
                elif "550b" in active_mod.lower():
                    judge_display = "NVIDIA Nemotron 3 Ultra 550B (OpenRouter)"
                elif "120b" in active_mod.lower():
                    judge_display = "NVIDIA Nemotron 3 Super 120B (OpenRouter)"
                elif "lightning" in active_mod.lower():
                    judge_display = "NVIDIA Nemotron 3.5 Lightning (OpenRouter)"
                else:
                    judge_display = f"{active_mod} (OpenRouter)"
                print(f"[Signal 2 LLM Judge] AI audit completed successfully using {judge_display}.")
            else:
                print("[Signal 2 LLM Judge] AI models exhausted or unavailable.")

        # Tier 2 local heuristic fallback has been removed: Signal 2 is strictly evaluated by the LLM
        if not llm_result:
            raise RuntimeError(
                "[Signal 2 Error] LLM Judge API unavailable or failed to parse response. "
                "Local Tier 2 heuristic engine has been removed per system specification. "
                "Please verify your API keys (OPENROUTER_API_KEY or GEMINI_API_KEY) in .env."
            )

        self.last_active_model = judge_display

        # Standardize and compute weighted metrics
        total_items = max(llm_result.get("total_items", len(llm_result.get("items", []))), 1)
        items = llm_result.get("items", [])

        # Normalize weights so they sum to 1.0
        raw_weights = [float(it.get("importance_weight", 0.5)) for it in items] or [1.0]
        total_w = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
        normalized_weights = [w / total_w for w in raw_weights]

        addressed_count = 0
        partial_count = 0
        missing_count = 0
        weighted_score_accum = 0.0

        item_breakdown = []
        for idx, it in enumerate(items):
            status = it.get("status", "NOT_PROVIDED_EVADED").upper()
            w = normalized_weights[idx]
            
            # Map LLM status to pipeline status standard
            if status in ["FURNISHED", "ADDRESSED", "PROVIDED"]:
                std_status = "ADDRESSED"
                score = 1.0
                addressed_count += 1
            elif status in ["PARTIALLY_FURNISHED", "PARTIALLY_ADDRESSED", "PARTIAL"]:
                std_status = "PARTIALLY_ADDRESSED"
                score = 0.5
                partial_count += 1
            elif "WITHHELD" in status or "EXEMPTION" in status:
                std_status = "WITHHELD"
                score = 0.0
                missing_count += 1
            else:
                std_status = "MISSING"
                score = 0.0
                missing_count += 1

            weighted_score_accum += (score * w)

            importance = it.get("importance", "CORE_OPERATIVE_DEMAND")
            if "CORE" in importance.upper():
                importance_label = "CORE_OPERATIVE_DEMAND"
            else:
                importance_label = "AUXILIARY_INQUIRY"

            item_breakdown.append({
                "item_index": idx + 1,
                "question_text": it.get("question_text", f"Query Item {idx+1}"),
                "importance": importance_label,
                "importance_weight": round(w, 3),
                "status": std_status,
                "raw_status": status,
                "compliance_score": score,
                "best_match_text": it.get("matched_reply_excerpt", "No substantive response on record."),
                "legal_reason": it.get("legal_reason", "Item compliance evaluated under RTI Act disclosure requirements."),
                "similarity_score": round(score, 2)  # For backwards-compatibility with existing UI meters
            })

        weighted_compliance_ratio = round(min(max(weighted_score_accum, 0.0), 1.0), 4)

        if weighted_compliance_ratio >= 0.85:
            verdict = "COMPLETE_COMPLIANCE"
        elif weighted_compliance_ratio >= 0.30:
            verdict = "PARTIAL_COMPLIANCE"
        else:
            verdict = "NON_COMPLIANT_EVASION"

        elapsed = round(time.time() - start_time, 3)

        # Build Explainable Plain-English Rationale
        rationale = self._generate_explainable_rationale(
            total_items=total_items,
            addressed=addressed_count,
            partial=partial_count,
            missing=missing_count,
            weighted_score=weighted_compliance_ratio,
            verdict=verdict,
            items=item_breakdown,
            summary=llm_result.get("audit_summary", "")
        )

        return {
            "case_id": case_id or "ANON_CASE",
            "execution_time_sec": elapsed,
            "judge_model": judge_display,
            "total_items": total_items,
            "addressed_count": addressed_count,
            "partial_count": partial_count,
            "missing_count": missing_count,
            "coverage_ratio": weighted_compliance_ratio,
            "compliance_verdict": verdict,
            "item_breakdown": item_breakdown,
            "explainable_rationale": rationale,
            "audit_summary": llm_result.get("audit_summary", "")
        }

    def _infer_llm(self, query: str, narrative: str) -> Optional[Dict]:
        """
        Executes prompt on local LLM and parses JSON output.
        """
        import torch

        user_content = (
            f"CITIZEN INFORMATION SOUGHT:\n{query}\n\n"
            f"PUBLIC AUTHORITY RESPONSE / CASE NARRATIVE:\n{narrative}\n\n"
            f"Perform the substantive compliance audit and output strictly the JSON object:"
        )

        messages = [
            {"role": "system", "content": JUDICIAL_AUDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,  # Greedy deterministic decoding
                temperature=None,
                top_p=None
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]
        response_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return self._extract_json(response_text)

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """
        Extracts and parses JSON from model generation, handling markdown fences and thinking blocks.
        """
        clean_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Remove ```json ... ``` wrapper if present
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if fence_match:
            clean_text = fence_match.group(1)
        else:
            # Fallback search for outer-most braces
            braces_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
            if braces_match:
                clean_text = braces_match.group(1)

        try:
            parsed = json.loads(clean_text)
            if isinstance(parsed, dict) and "items" in parsed:
                return parsed
        except Exception:
            pass
        return None

    @staticmethod
    def _generate_explainable_rationale(
        total_items: int,
        addressed: int,
        partial: int,
        missing: int,
        weighted_score: float,
        verdict: str,
        items: List[Dict],
        summary: str
    ) -> str:
        """
        Synthesizes an explainable, judicial compliance rationale for the bench.
        """
        parts = []
        parts.append(
            f"Signal 2 Compliance Audit: Dissected citizen query into {total_items} itemized request(s). "
            f"Factual analysis reveals {addressed} fully furnished, {partial} partially addressed, and {missing} unaddressed/withheld, "
            f"yielding a weighted substantive compliance score of {weighted_score*100:.1f}% ({verdict.replace('_', ' ')})."
        )

        # Highlight missing core demands
        missing_core = [
            f"Q{it['item_index']} ('{it['question_text'][:40]}...')"
            for it in items if it["status"] in ["MISSING", "WITHHELD"] and it["importance"] == "CORE_OPERATIVE_DEMAND"
        ]
        if missing_core:
            parts.append(
                f"Statutory Notice: Core operative demands ({', '.join(missing_core)}) were not substantively furnished by the PIO, "
                f"triggering a disclosure mandate under Section 7(1) of the RTI Act."
            )

        if summary:
            parts.append(f"Judicial Auditor Finding: {summary}")

        return " ".join(parts)

    def generate_case_finding(
        self,
        citizen_query: str,
        pio_reply: str,
        final_outcome: str,
        governing_clause: Optional[str] = "RTI Act Statutory Provisions",
        days_elapsed: int = 15,
        exemptions_cited: str = "",
        item_breakdown: Optional[List[Dict]] = None,
        rule_triggered: Optional[str] = None
    ) -> str:
        """
        Directly asks the LLM to explain why the pipeline's predicted outcome label
        was chosen given the Citizen's Query, the PIO's Reply, and the Label.
        """
        import torch

        outcome_display = final_outcome.replace('_', ' ').title()

        # Fallback if LLM is not responding or unavailable
        fallback = "LLM not working"

        system_prompt = (
            "You are a Senior Judicial Registrar and Legal Research Officer analyzing Right to Information (RTI) Act, 2005 disputes.\n\n"
            "Task: You are given three inputs: (1) the RTI Query filed by the applicant, (2) the Response given by the PIO, "
            f"and (3) the Predicted Output (the judgment label '{outcome_display}') produced by an upstream system.\n\n"
            "Your job is not to decide, verify or change the predicted output. Treat it as fixed and correct. "
            "Your only task is to write a short Judicial Reasoning that explains why this predicted output follows from the specific Query and Response.\n\n"
            "To do this, first identify what exactly the applicant asked for, then identify what exactly the PIO did "
            "(provided the information, partially provided it, rejected it, cited an exemption, cited no exemption, gave no reasons, or missed the deadline). "
            "Then connect those facts to the predicted output using only the RTI Act provisions that genuinely apply to those facts.\n\n"
            "Strict Adjudication Rules:\n"
            "- Base every statement strictly on what appears in the Query and Response.\n"
            "- Do not invent facts, do not assume unanswered questions or missing information unless the Response clearly shows it.\n"
            "- Do not cite sections that do not match the PIO's actual conduct (for example, cite Section 7(2) only for a missed deadline or no reply, not for an express rejection).\n"
            "- Write 3 to 4 sentences in formal judicial tone, strictly in this format:\n"
            f'  "{outcome_display} was chosen because [what the PIO did or failed to do, in relation to what was asked]. [Relevant legal provisions and why they apply]. Hence [consequence/direction]."'
        )

        user_content = (
            f"1. RTI Query Filed by Applicant:\n{citizen_query.strip()}\n\n"
            f"2. Response Given by PIO:\n{pio_reply.strip()}\n\n"
            f"3. Predicted Output (Fixed Judgment Label):\n{outcome_display}\n\n"
            f"Write the 3 to 4 sentence Judicial Reasoning in the exact specified format:"
        )

        # 1. Try OpenRouter API first across configured API keys
        api_keys = self._get_api_keys_to_try()
        if api_keys:
            url = "https://openrouter.ai/api/v1/chat/completions"

            for key_idx, key in enumerate(api_keys):
                headers = {
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost:5173",
                    "X-Title": "RTI Neuro-Symbolic Judicial Auditor"
                }
                key_mask = f"...{key[-6:]}" if len(key) > 6 else "key"

                for candidate_model in self._get_models_to_try():
                    payload = {
                        "model": candidate_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        "temperature": 0.1,
                        "max_tokens": 800
                    }
                    try:
                        req = urllib.request.Request(
                            url,
                            data=json.dumps(payload).encode("utf-8"),
                            headers=headers,
                            method="POST"
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp:
                            resp_data = json.loads(resp.read().decode("utf-8"))
                            choices = resp_data.get("choices")
                            if choices and len(choices) > 0 and "message" in choices[0]:
                                msg = choices[0]["message"]
                                raw_finding = (msg.get("content") or "").strip()
                                cleaned = self._clean_case_finding(raw_finding, outcome_display)
                                if cleaned:
                                    print(f"[Signal 2 OpenRouter] Generated Judicial Reasoning using model: {candidate_model} (key {key_mask})")
                                    self.active_reasoning_model = candidate_model
                                    return cleaned
                            elif "error" in resp_data:
                                err_msg = resp_data["error"].get("message", "error")
                                code = resp_data["error"].get("code", 0)
                                print(f"[Signal 2 OpenRouter] {candidate_model} (key {key_mask}) reasoning error: {err_msg[:60]}")
                                if code == 429 or "rate limit" in err_msg.lower():
                                    print(f"[Signal 2 OpenRouter] Rate limit reached on key {key_mask}; failing over to next API key...")
                                    break
                    except urllib.error.HTTPError as e:
                        err_body = e.read().decode("utf-8", errors="ignore")
                        print(f"[Signal 2 OpenRouter] {candidate_model} (key {key_mask}) HTTP {e.code} ({err_body[:60]})")
                        if e.code == 429:
                            print(f"[Signal 2 OpenRouter] HTTP 429 on key {key_mask}; failing over to next API key...")
                            break
                    except Exception as e:
                        print(f"[Signal 2 OpenRouter] {candidate_model} reasoning request failed ({e}); cascading to next model...")

        # 2. Try Google AI API (Gemini / Gemma) if OpenRouter rate limit reached
        google_raw = self._infer_google_api(system_prompt, user_content, max_tokens=600)
        if google_raw:
            cleaned = self._clean_case_finding(google_raw, outcome_display)
            if cleaned:
                print(f"[Signal 2 Google AI] Generated Judicial Reasoning via Google AI.")
                self.active_reasoning_model = "Google AI (Gemma/Gemini)"
                return cleaned

        # 3. Try local model if loaded
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            try:
                import torch
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=180,
                        do_sample=False,
                        temperature=None,
                        top_p=None
                    )

                generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
                finding_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                if "Case Finding:" in finding_text:
                    finding_text = finding_text.split("Case Finding:")[-1].strip()
                finding_text = finding_text.replace("\n", " ").strip()

                if len(finding_text) > 20:
                    return finding_text
            except Exception as e:
                print(f"[Signal 2 LLM Judge] Local case finding error: {e}")

        return fallback

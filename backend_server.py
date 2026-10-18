"""
RTI Neuro-Symbolic API Server
Serves the complete 3-Signal Master Decision Making Engine to the React frontend.
Port: 8000
"""

import os
import sys
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

from dotenv import load_dotenv

# Setup project root
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

load_dotenv(os.path.join(current_dir, ".env"))

from decision_engine.master_engine import NeuroSymbolicDecisionEngine, DecisionVerdict

print("=" * 80)
print(" RTI NEURO-SYMBOLIC MASTER API SERVER (BACKEND SERVICE)")
print(" Loading Master Decision Making Engine (Signal 1 + Signal 2 + Signal 3)...")
print("=" * 80)

engine = None
try:
    engine = NeuroSymbolicDecisionEngine()
    print("[OK] Master Decision Engine loaded successfully!")
except Exception as e:
    print(f"[ERROR] Failed to load Master Decision Engine: {e}")


def format_frontend_payload(verdict: DecisionVerdict, data: dict, raw_s2: dict, s3_facts) -> dict:
    """
    Transforms DecisionVerdict into the exact schema consumed by React frontend cards.
    """
    citizen_query = data.get("citizen_query") or data.get("citizenQuery", "")
    pio_reply = data.get("pio_reply") or data.get("pioReply", "")
    authority = data.get("public_authority") or data.get("publicAuthority", "Public Authority")
    exemptions = data.get("exemptions_cited") or data.get("exemptionsCited", [])
    raw_days = data.get("days_elapsed") if data.get("days_elapsed") is not None else data.get("daysElapsed", 15)
    try:
        days_elapsed = int(raw_days)
    except (ValueError, TypeError):
        days_elapsed = 15

    # 1. Format Signal 1 Data (InLegalBERT Domain Neural Classifier)
    s1_raw = getattr(verdict, "raw_signal1_result", None) or {}
    s1_class = s1_raw.get("predicted_class") or verdict.raw_signal_predictions.get("signal_1_neural", "INFORMATION_DIRECTED")
    s1_probs = s1_raw.get("probabilities") or verdict.fused_probabilities
    s1_conf = round(float(s1_raw.get("confidence", s1_probs.get(s1_class, 0.75))) * 100)

    signal1_data = {
        "predictedClass": s1_class,
        "confidence": s1_conf,
        "probabilities": s1_probs,
        "modelName": "InLegalBERT (73.1% Acc / 73.4% F1)",
        "isLiveModel": True,
        "device": str(getattr(engine.signal1, "device", "CUDA")),
        "tokensProcessed": min(512, max(24, int((len(citizen_query) + len(pio_reply)) / 3.8))),
        "headTokens": 160,
        "tailTokens": 350
    }

    # 2. Format Signal 2 Data
    metrics = raw_s2.get("coverage_metrics", {})
    item_breakdown = raw_s2.get("item_breakdown", [])
    formatted_items = []
    for idx, item in enumerate(item_breakdown):
        st = item.get("status", "ADDRESSED")
        badge = "badge-addressed"
        label = "Furnished"
        if st == "PARTIALLY_ADDRESSED":
            badge = "badge-partial"
            label = "Partially Furnished"
        elif st == "WITHHELD":
            badge = "badge-withheld"
            label = "Withheld (Exemption)"
        elif st in ["MISSING", "NOT_PROVIDED_EVADED"]:
            badge = "badge-missing"
            label = "Evaded / Not Furnished"

        formatted_items.append({
            "queryIndex": idx + 1,
            "queryText": item.get("question_text", f"Query Item {idx+1}"),
            "status": st,
            "statusLabel": label,
            "badgeClass": badge,
            "matchedReplyContext": item.get("best_match_text", "No record in response"),
            "cosineSimilarity": float(item.get("compliance_score", item.get("similarity_score", 0.0))),
            "importance": item.get("importance", "CORE_OPERATIVE_DEMAND"),
            "importanceWeight": float(item.get("importance_weight", 0.5)),
            "complianceScore": float(item.get("compliance_score", 0.0)),
            "legalReason": item.get("legal_reason", "")
        })

    signal2_data = {
        "totalItems": metrics.get("total_items", len(formatted_items) or 1),
        "addressedCount": metrics.get("addressed_count", 0),
        "partialCount": metrics.get("partial_count", 0),
        "missingCount": metrics.get("missing_count", 0),
        "coverageRatio": metrics.get("coverage_ratio", 0.0),
        "complianceVerdict": metrics.get("compliance_verdict", "UNKNOWN"),
        "evasionRate": round((1.0 - metrics.get("coverage_ratio", 1.0)) * 100),
        "items": formatted_items,
        "embeddingModel": raw_s2.get("judge_model", "NVIDIA Nemotron 3 Ultra 550B (LLM Judge)"),
        "judgeModel": raw_s2.get("judge_model", "NVIDIA Nemotron 3 Ultra 550B (LLM Judge)"),
        "extractionModel": "LLM Clause Decomposer + Substantive Compliance Auditor"
    }

    # 3. Format Signal 3 Data
    detected_sections = []
    for sec in getattr(s3_facts, "exemptions_detected", []):
        sec_clean = str(sec).strip()
        title = "Statutory Exemption"
        if sec_clean == "8(1)(j)":
            title = "Personal Privacy & 3rd-Party Data"
        elif sec_clean == "8(1)(d)":
            title = "Commercial Confidence & Trade Secrets"
        elif sec_clean == "8(1)(e)":
            title = "Fiduciary Relationship"
        elif sec_clean == "8(1)(h)":
            title = "Pending Investigation / Prosecution"
        elif sec_clean == "8(1)(a)":
            title = "National Security & Sovereignty"

        detected_sections.append({
            "section": f"Sec {sec_clean}",
            "title": title,
            "detail": f"Statutory bar detected in submissions for Section {sec_clean}.",
            "severity": "high"
        })

    if getattr(s3_facts, "is_sec24_organization", False):
        detected_sections.append({
            "section": "Sec 24",
            "title": "Second Schedule Intelligence Agency",
            "detail": f"Public Authority ({getattr(s3_facts, 'sec24_agency_name', 'Security Agency')}) is exempt from RTI jurisdiction.",
            "severity": "critical"
        })

    if getattr(s3_facts, "sec20_penalty_triggered", False):
        detected_sections.append({
            "section": "Sec 20",
            "title": "Statutory Penal Sanction Threshold",
            "detail": "Show-Cause grounds detected: Unreasonable delay or malafide obstruction.",
            "severity": "critical"
        })

    if getattr(s3_facts, "sec10_severability_indicated", False):
        detected_sections.append({
            "section": "Sec 10",
            "title": "Severability Principle",
            "detail": "Mixed disclosure indicated: non-exempt portion to be severed.",
            "severity": "medium"
        })

    # Build dynamic jurisprudential precedents based on detected flags
    precedents_matched = []
    if getattr(s3_facts, "is_sec24_organization", False):
        precedents_matched.append({
            "citation": "D.G. Income Tax (Inv.) v. Sudhir Kumar (CIC/WB/A/2009/00024)",
            "principle": "Section 24 complete jurisdictional immunity unless human rights or corruption allegations are substantiated."
        })
    if getattr(s3_facts, "has_valid_sec8_claim", False):
        sec8_clauses = getattr(s3_facts, "cited_sec8_clauses", [])
        if any("8(1)(j)" in c or "privacy" in c for c in sec8_clauses):
            precedents_matched.append({
                "citation": "Girish Ramchandra Deshpande v. CIC (2013) 1 SCC 212",
                "principle": "Personal information and service records protected under Section 8(1)(j) in absence of larger public interest."
            })
        if any("8(1)(d)" in c for c in sec8_clauses):
            precedents_matched.append({
                "citation": "CBSE v. Aditya Bandopadhyay (2011) 8 SCC 497",
                "principle": "Internal evaluation methodology & commercial confidences protected under Section 8(1)(d)."
            })
        if not precedents_matched:
            precedents_matched.append({
                "citation": "CBSE v. Aditya Bandopadhyay (2011) 8 SCC 497",
                "principle": "Statutory boundaries of disclosure and fiduciary limits under Section 8(1)."
            })
    if getattr(s3_facts, "sec20_penalty_triggered", False):
        precedents_matched.append({
            "citation": "Union of India v. Vishwas Bhamburkar (Delhi High Court W.P.(C) 3660/2012)",
            "principle": "Plea of untraceable records or unexcused delay exceeds reasonable cause; Section 20 show-cause mandatory."
        })
    if getattr(s3_facts, "sec10_severability_indicated", False):
        precedents_matched.append({
            "citation": "CIC/SG/A/2012/000214",
            "principle": "Severability under Section 10 strictly enforced to furnish non-exempt portions."
        })
    if not precedents_matched:
        precedents_matched.append({
            "citation": "State of UP v. Raj Narain (AIR 1975 SC 865)",
            "principle": "Citizens have a constitutional right to know public actions under Article 19(1)(a); disclosure is the norm."
        })

    signal3_data = {
        "detectedSections": detected_sections,
        "flags": {
            "is_sec8_claimed": getattr(s3_facts, "has_valid_sec8_claim", False),
            "is_sec24_claimed": getattr(s3_facts, "is_sec24_organization", False),
            "is_sec20_penalty_risk": getattr(s3_facts, "sec20_penalty_triggered", False),
            "is_severability_indicated": getattr(s3_facts, "sec10_severability_indicated", False)
        },
        "statutoryFramework": "RTI Act, 2005 (Formal Legal Ontology)",
        "precedentsMatched": precedents_matched
    }

    # 4. Format Stage 4 Data
    case_finding = None
    if hasattr(engine, "signal2") and hasattr(engine.signal2, "generate_case_finding"):
        try:
            case_finding = engine.signal2.generate_case_finding(
                citizen_query=citizen_query,
                pio_reply=pio_reply,
                days_elapsed=days_elapsed,
                exemptions_cited=str(exemptions),
                final_outcome=verdict.final_predicted_outcome,
                governing_clause=verdict.governing_statutory_clause,
                item_breakdown=raw_s2.get("item_breakdown", []),
                rule_triggered=verdict.symbolic_rule_triggered
            )
        except Exception as e:
            print(f"[Backend Server] Case finding generation error: {e}")
            case_finding = "LLM not working"

    if not case_finding:
        case_finding = "LLM not working"

    stage4_data = {
        "finalOutcome": verdict.final_predicted_outcome,
        "certaintyScore": round(verdict.confidence * 100),
        "overrideApplied": verdict.is_statutory_override,
        "ruleTriggered": verdict.symbolic_rule_triggered,
        "governingClause": verdict.governing_statutory_clause,
        "legalReasoningTrace": verdict.reasoning_audit_trace,
        "judicialDirections": [verdict.actionable_judicial_directive],
        "caseFinding": case_finding,
        "fusedProbabilities": verdict.fused_probabilities,
        "contributions": verdict.contributions_by_signal,
        "rawSignalVotes": verdict.raw_signal_predictions,
        "latency": verdict.execution_time_sec
    }

    return {
        "timestamp": str(os.environ.get("TIMESTAMP", "2026-09-22T19:00:00Z")),
        "inputs": {
            "citizenQuery": citizen_query,
            "pioReply": pio_reply,
            "publicAuthority": authority,
            "exemptionsCited": exemptions,
            "daysElapsed": days_elapsed
        },
        "signal1": signal1_data,
        "signal2": signal2_data,
        "signal3": signal3_data,
        "stage4": stage4_data
    }


class SimpleRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Access-Control-Max-Age', '86400')

    def _send_json(self, status_code: int, data: dict):
        try:
            body = json.dumps(data).encode('utf-8')
        except Exception:
            body = b'{}'
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Content-Length', '0')
        self.send_header('Connection', 'close')
        self._send_cors_headers()
        self.end_headers()
        self.close_connection = True

    def do_GET(self):
        response = {
            "status": "online",
            "service": "Master RTI Neuro-Symbolic Decision Making Engine",
            "engine_loaded": engine is not None,
            "device": str(getattr(engine.signal1, "device", "CUDA")) if engine else "unavailable",
            "weights": {
                "signal1_neural": 0.50,
                "signal2_coverage": 0.30,
                "signal3_statutory": 0.20
            }
        }
        self._send_json(200, response)

    def do_POST(self):
        if self.path in ["/api/decision_engine", "/api/predict", "/api/predict_signal_1"]:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            try:
                data = json.loads(post_data) if post_data else {}
            except Exception:
                data = {}

            citizen_query = data.get("citizen_query") or data.get("citizenQuery") or data.get("information_sought") or data.get("informationSought", "")
            pio_reply = data.get("pio_reply") or data.get("pioReply") or data.get("case_narrative") or data.get("caseNarrative", "")
            exemptions = data.get("exemptions_cited") or data.get("exemptionsCited", [])
            authority = data.get("public_authority") or data.get("publicAuthority", "Public Authority")

            if isinstance(exemptions, list):
                exemptions_str = json.dumps(exemptions)
            else:
                exemptions_str = str(exemptions)

            raw_days = data.get("days_elapsed") if data.get("days_elapsed") is not None else data.get("daysElapsed", 15)
            try:
                days_elapsed = int(raw_days)
            except (ValueError, TypeError):
                days_elapsed = 15
            data["days_elapsed"] = days_elapsed

            if engine is not None:
                try:
                    # Run full Master Decision Engine
                    verdict: DecisionVerdict = engine.decide(
                        case_narrative=pio_reply,
                        information_sought=citizen_query,
                        public_authority=authority,
                        exemptions_cited=exemptions_str,
                        case_id="WEB_UI_QUERY"
                    )

                    # Get raw Signal 2 & Signal 3 objects for rich UI formatting (cached from decide)
                    raw_s2 = verdict.raw_signal2_coverage if verdict.raw_signal2_coverage is not None else engine.signal2.analyze_case(
                        information_sought=citizen_query,
                        case_narrative=pio_reply,
                        case_id="WEB_UI_QUERY"
                    )
                    s3_facts = verdict.raw_signal3_facts if verdict.raw_signal3_facts is not None else engine.scanner.scan_case(
                        case_narrative=pio_reply,
                        information_sought=citizen_query,
                        public_authority=authority,
                        exemptions_cited=exemptions_str,
                        case_id="WEB_UI_QUERY"
                    )

                    # Build complete React-ready payload
                    response_payload = format_frontend_payload(verdict, data, raw_s2, s3_facts)
                    self._send_json(200, response_payload)
                    return
                except Exception as e:
                    print(f"[Backend Server] Engine decision exception: {e}")
                    self._send_json(500, {"error": str(e)})
                    return
            else:
                self._send_json(503, {"error": "Decision Engine not loaded"})
                return

        self._send_json(404, {"error": "Not Found"})

    def log_message(self, format, *args):
        sys.stderr.write(f"[Decision Engine Server] {args[0]} - {args[1]} {args[2]}\n")


def run_server(port=8000):
    server_address = ('0.0.0.0', port)
    httpd = ThreadingHTTPServer(server_address, SimpleRequestHandler)
    print(f"\n[OK] Decision Engine API Server running at http://127.0.0.1:{port}/ and http://localhost:{port}/")
    print(f"Ready to serve live AI predictions to the React UI.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)

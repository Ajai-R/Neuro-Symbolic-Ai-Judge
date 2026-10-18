/**
 * Unified API Service Layer
 * 
 * Interacts with the Python backend server (backend_server.py on port 8000).
 * Retrieves real live PyTorch predictions from the Master Neuro-Symbolic Decision Making Engine
 * (Signal 1 InLegalBERT, Signal 2 SentenceTransformers, Signal 3 Statutory Scanner, and Decision Synthesis).
 */

import { OUTCOME_META } from "./pipelineEngine";

const getBackendUrl = () => {
  if (typeof window !== 'undefined' && window.location.hostname) {
    return `http://${window.location.hostname}:8000`;
  }
  return 'http://127.0.0.1:8000';
};

export async function checkBackendHealth() {
  const primaryUrl = getBackendUrl();
  const candidateUrls = [primaryUrl, "http://127.0.0.1:8000", "http://localhost:8000"];
  const urls = [...new Set(candidateUrls)];

  for (const url of urls) {
    try {
      const res = await fetch(`${url}/`, { method: "GET" });
      if (res.ok) {
        const data = await res.json();
        return {
          status: "online",
          model_loaded: Boolean(data.engine_loaded || data.model_loaded),
          device: data.device || "CUDA",
          service: data.service,
          url: url
        };
      }
    } catch (e) {
      // try next fallback
    }
  }
  return { status: "offline", model_loaded: false };
}

export async function predictRTIOutcome(payload) {
  const requestBody = JSON.stringify({
    citizen_query: payload.citizenQuery,
    pio_reply: payload.pioReply,
    exemptions_cited: payload.exemptionsCited,
    public_authority: payload.publicAuthority,
    days_elapsed: payload.daysElapsed
  });

  const primaryUrl = getBackendUrl();
  const candidateUrls = [primaryUrl, "http://127.0.0.1:8000", "http://localhost:8000"];
  const urls = [...new Set(candidateUrls)];

  let lastError = null;
  let response = null;

  for (const baseUrl of urls) {
    try {
      response = await fetch(`${baseUrl}/api/decision_engine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestBody
      });
      if (response && response.ok) {
        break;
      }
    } catch (err) {
      lastError = err;
    }
  }

  if (!response || !response.ok) {
    throw new Error(
      lastError
        ? `Backend connection error: ${lastError.message}. Please verify python backend_server.py is running on port 8000.`
        : `Backend server returned HTTP ${response?.status}. Please verify python backend_server.py is running on port 8000.`
    );
  }

  const liveData = await response.json();
  console.log("[Decision Engine] Live AI Response:", liveData);

  // Attach outcomeMeta for styling
  if (liveData.stage4 && !liveData.stage4.outcomeMeta) {
    liveData.stage4.outcomeMeta = OUTCOME_META[liveData.stage4.finalOutcome] || {};
  }

  return liveData;
}

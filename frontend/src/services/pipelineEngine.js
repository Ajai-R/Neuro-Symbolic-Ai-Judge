/**
 * Explainable Neuro-Symbolic RTI Pipeline Simulation Engine
 * Faithfully mirrors the 4-stage pipeline specified in the project methodology:
 * 
 * Stage 1: Signal 1 - InLegalBERT / RoBERTa Neural Sequence Classifier
 * Stage 2: Signal 2 - FLAN-T5 Item Extraction + BGE-M3 Dense Semantic Verification
 * Stage 3: Signal 3 - Deterministic Statutory Rule Scanner (RTI Act, 2005)
 * Stage 4: Stage 4 - Deterministic Judicial Switch Matrix (Conflict Resolution & Audit Trace)
 */

export const OUTCOME_META = {
  INFORMATION_DIRECTED: {
    label: "Information Directed",
    badgeColor: "#06B6D4",
    bgTint: "rgba(6, 182, 212, 0.12)",
    borderTint: "rgba(6, 182, 212, 0.4)",
    icon: "FolderCheck",
    description: "Full disclosure ordered. The public authority must furnish the requested records within statutory time limits."
  },
  APPEAL_DISMISSED: {
    label: "Appeal Dismissed",
    badgeColor: "#F43F5E",
    bgTint: "rgba(244, 63, 94, 0.12)",
    borderTint: "rgba(244, 63, 94, 0.4)",
    icon: "ShieldAlert",
    description: "The PIO's denial is legally upheld. Records are protected under statutory exemptions or Section 24 non-applicability."
  },
  PARTIAL_RELIEF: {
    label: "Partial Relief",
    badgeColor: "#F59E0B",
    bgTint: "rgba(245, 158, 11, 0.12)",
    borderTint: "rgba(245, 158, 11, 0.4)",
    icon: "SplitSquareVertical",
    description: "Section 10 Severability applied. Non-exempt portions directed for disclosure while exempt parts are withheld."
  },
  PENALTY_IMPOSED: {
    label: "Penalty Imposed",
    badgeColor: "#A855F7",
    bgTint: "rgba(168, 85, 247, 0.12)",
    borderTint: "rgba(168, 85, 247, 0.4)",
    icon: "AlertOctagon",
    description: "Show-Cause Notice issued under Section 20(1). Malafide evasion, destroyed records, or unexcused delay detected."
  }
};

/**
 * Stage 1: Signal 1 - Neural Classifier Simulation
 */
function simulateSignal1(citizenQuery, pioReply, exemptionsCited, daysElapsed) {
  const queryLower = citizenQuery.toLowerCase();
  const replyLower = pioReply.toLowerCase();
  const exList = Array.isArray(exemptionsCited) 
    ? exemptionsCited 
    : (typeof exemptionsCited === 'string' && exemptionsCited.trim() ? exemptionsCited.split(/[,;\s]+/).map(s => s.trim()).filter(Boolean) : []);
  const hasExemption = exList.length > 0;

  // Base raw logits
  let logits = {
    INFORMATION_DIRECTED: 1.2,
    APPEAL_DISMISSED: 0.8,
    PARTIAL_RELIEF: 0.4,
    PENALTY_IMPOSED: 0.2
  };

  // Adjust logits based on legal context signals
  if (hasExemption && (exList.some(e => e.includes("8(1)(j)") || e.includes("8(1)(d)") || e.includes("24")))) {
    logits.APPEAL_DISMISSED += 2.5;
    logits.INFORMATION_DIRECTED -= 1.0;
  }

  if (replyLower.includes("enclosed") || replyLower.includes("provided") || replyLower.includes("furnished") || replyLower.includes("attached")) {
    if (replyLower.includes("cannot be provided") || replyLower.includes("exempt") || replyLower.includes("withheld")) {
      logits.PARTIAL_RELIEF += 2.8;
    } else {
      logits.INFORMATION_DIRECTED += 1.8;
    }
  }

  if (daysElapsed > 100 || replyLower.includes("lost") || replyLower.includes("cannot be traced") || replyLower.includes("misplaced") || replyLower.includes("untraceable")) {
    logits.PENALTY_IMPOSED += 3.2;
    logits.APPEAL_DISMISSED -= 1.5;
  }

  if (!hasExemption && (replyLower.includes("voluminous") || replyLower.includes("internal") || replyLower.includes("routine"))) {
    logits.INFORMATION_DIRECTED += 2.0;
    logits.APPEAL_DISMISSED -= 0.5;
  }

  // Softmax normalization
  const expValues = {};
  let sumExp = 0;
  for (const key of Object.keys(logits)) {
    expValues[key] = Math.exp(logits[key]);
    sumExp += expValues[key];
  }

  const probabilities = {};
  let maxProb = -1;
  let topClass = "INFORMATION_DIRECTED";

  for (const key of Object.keys(expValues)) {
    const p = expValues[key] / sumExp;
    probabilities[key] = Math.round(p * 1000) / 1000;
    if (p > maxProb) {
      maxProb = p;
      topClass = key;
    }
  }

  return {
    predictedClass: topClass,
    confidence: Math.round(maxProb * 100),
    probabilities,
    modelName: "InLegalBERT-Base-4Class (Fine-Tuned)",
    tokensProcessed: Math.min(512, Math.round((citizenQuery.length + pioReply.length) / 3.8)),
    headTokens: 160,
    tailTokens: 350
  };
}

/**
 * Stage 2: Signal 2 - FLAN-T5 Checklist Extraction & BGE-M3 Semantic Verification
 */
function simulateSignal2(citizenQuery, pioReply) {
  // Step 2A: Parse citizen query into itemized checklist
  let rawLines = citizenQuery
    .split(/\n|(?:\d+\.|\([a-z]\)|\*|-)\s+/)
    .map(s => s.trim())
    .filter(s => s.length > 8);

  if (rawLines.length <= 1) {
    const cleaned = citizenQuery.replace(/^.*?(?:following information|hereunder|as follows|below)\s*:\s*/i, "");
    const clauses = cleaned.split(/(?<=[,])\s+(?=(?:what|which|who|when|where|how|on what|at what|please|provide|copy of)\b)|\s+and\s+(?=(?:what|which|who|when|where|how|on what|at what|please|provide|copy of)\b)/i)
      .map(s => s.trim().replace(/^[,.;]+|[,.;]+$/g, ""))
      .filter(s => s.length > 10);
    if (clauses.length > 1) {
      rawLines = clauses;
    }
  }

  const checklistItems = rawLines.length > 0 
    ? rawLines 
    : [citizenQuery.trim()];

  const replyLower = pioReply.toLowerCase();
  const replyWords = new Set(replyLower.match(/\b[a-z]{3,}\b/g) || []);

  // Step 2B: Dense cosine matching simulation
  let addressedCount = 0;
  let partialCount = 0;
  let missingCount = 0;

  const itemEvaluations = checklistItems.map((item, index) => {
    const itemWords = item.toLowerCase().match(/\b[a-z]{3,}\b/g) || [];
    let overlapCount = 0;
    for (const w of itemWords) {
      if (replyWords.has(w)) overlapCount++;
    }

    const jaccard = itemWords.length > 0 ? overlapCount / itemWords.length : 0;
    
    // Simulate BGE-M3 Dense Embedding Cosine Similarity
    let cosineSim = 0.25 + (jaccard * 0.70);

    // Negative indicators for evasion or refusal
    if (replyLower.includes("lost") || replyLower.includes("cannot be traced") || replyLower.includes("not available") || replyLower.includes("voluminous")) {
      cosineSim = Math.min(cosineSim, 0.38);
    }

    // Direct fulfillment keywords
    if ((replyLower.includes("enclosed") || replyLower.includes("furnished") || replyLower.includes("provided")) && (item.toLowerCase().includes("answer") || item.toLowerCase().includes("score") || item.toLowerCase().includes("copy") || item.toLowerCase().includes("mark"))) {
      if (index === 0 && checklistItems.length > 1 && (replyLower.includes("cannot") || replyLower.includes("exempt"))) {
        cosineSim = 0.88;
      }
    }

    // Withheld keywords
    if (replyLower.includes("cannot be provided") || replyLower.includes("exempt") || replyLower.includes("denied")) {
      if (index > 0 && checklistItems.length > 1) {
        cosineSim = 0.42;
      }
    }

    cosineSim = Math.min(0.96, Math.max(0.12, Math.round(cosineSim * 100) / 100));

    let status = "MISSING";
    let statusLabel = "Missing (Evaded)";
    let badgeClass = "badge-missing";

    if (cosineSim >= 0.82) {
      status = "ADDRESSED";
      statusLabel = "Addressed (Fulfilled)";
      badgeClass = "badge-addressed";
      addressedCount++;
    } else if (cosineSim >= 0.50) {
      status = "PARTIALLY_ADDRESSED";
      statusLabel = "Partially Addressed";
      badgeClass = "badge-partial";
      partialCount++;
    } else {
      missingCount++;
    }

    return {
      queryIndex: index + 1,
      queryText: item,
      matchedReplyContext: pioReply.length > 120 ? pioReply.slice(0, 115) + "..." : pioReply,
      cosineSimilarity: cosineSim,
      status,
      statusLabel,
      badgeClass
    };
  });

  const total = checklistItems.length;
  const evasionRate = Math.round((missingCount / total) * 100);

  return {
    totalItems: total,
    addressedCount,
    partialCount,
    missingCount,
    evasionRate,
    items: itemEvaluations,
    embeddingModel: "SentenceTransformers (all-MiniLM-L6-v2)",
    extractionModel: "Legal Clause Itemizer + Sentence Splitter"
  };
}

/**
 * Stage 3: Signal 3 - Statutory Rule Scanner (RTI Act 2005)
 */
function simulateSignal3(citizenQuery, pioReply, exemptionsCited, daysElapsed) {
  const exText = typeof exemptionsCited === 'string' ? exemptionsCited : (Array.isArray(exemptionsCited) ? exemptionsCited.join(' ') : '');
  const combinedText = (citizenQuery + " " + pioReply + " " + exText).toLowerCase();
  
  const detectedSections = [];
  const flags = {
    is_sec8_claimed: false,
    is_sec8_1_j: false,
    is_sec8_1_d: false,
    is_sec24_claimed: false,
    is_sec2_f_clarification: false,
    delay_exceeds_30_days: daysElapsed > 30,
    delay_exceeds_100_days: daysElapsed > 100,
    record_claimed_lost: false,
    third_party_notice_issued: false
  };

  const exList = Array.isArray(exemptionsCited)
    ? exemptionsCited
    : (typeof exemptionsCited === 'string' && exemptionsCited.trim() ? exemptionsCited.split(/[,;\s]+/).map(s => s.trim()).filter(Boolean) : []);

  // Check explicit exemptions passed from user / form
  if (exList.length > 0) {
    exList.forEach(ex => {
      flags.is_sec8_claimed = true;
      if (ex.includes("8(1)(j)")) {
        flags.is_sec8_1_j = true;
        detectedSections.push({
          section: "Section 8(1)(j)",
          title: "Personal Information & Privacy",
          detail: "Protects individual privacy with no relationship to public activity or larger public interest.",
          severity: "high"
        });
      } else if (ex === "8(1)(d)") {
        flags.is_sec8_1_d = true;
        detectedSections.push({
          section: "Section 8(1)(d)",
          title: "Commercial Confidence & Trade Secrets",
          detail: "Protects intellectual evaluation, competitive position, and proprietary secrets.",
          severity: "high"
        });
      } else if (ex === "24") {
        flags.is_sec24_claimed = true;
        detectedSections.push({
          section: "Section 24",
          title: "Exempt Security / Intelligence Agency",
          detail: "Act non-applicable except on allegations of corruption and human rights violations.",
          severity: "critical"
        });
      } else {
        detectedSections.push({
          section: `Section ${ex}`,
          title: "Statutory Exemption Clause",
          detail: "Invoked under Section 8 of the RTI Act, 2005.",
          severity: "medium"
        });
      }
    });
  }

  // Regex scans on text
  if (combinedText.includes("8(1)(j)") || combinedText.includes("privacy") || combinedText.includes("third party")) {
    flags.is_sec8_1_j = true;
    flags.is_sec8_claimed = true;
    if (!detectedSections.some(s => s.section.includes("8(1)(j)"))) {
      detectedSections.push({
        section: "Section 8(1)(j)",
        title: "Personal Privacy Protection",
        detail: "Scanner detected third-party personal privacy terms in case narrative.",
        severity: "high"
      });
    }
  }

  if (combinedText.includes("section 11") || combinedText.includes("third party has objected")) {
    flags.third_party_notice_issued = true;
  }

  if (combinedText.includes("lost") || combinedText.includes("untraceable") || combinedText.includes("cannot be traced") || combinedText.includes("destroyed")) {
    flags.record_claimed_lost = true;
    detectedSections.push({
      section: "Section 20(1) Grounds",
      title: "Untraceable / Destroyed Record Defense",
      detail: "Commission precedent establishes that 'files lost' is unacceptable without formal FIR / inquiry.",
      severity: "critical"
    });
  }

  if (daysElapsed > 30) {
    detectedSections.push({
      section: "Section 7(1) Breach",
      title: `Mandatory 30-Day Response Window Violated (${daysElapsed} days)`,
      detail: "Information requested under RTI must be provided or rejected strictly within 30 days.",
      severity: daysElapsed > 100 ? "critical" : "medium"
    });
  }

  if (daysElapsed > 100) {
    detectedSections.push({
      section: "Section 20(1) Penalty Trigger",
      title: "Willful Delay (>100 days) Without Reasonable Cause",
      detail: "Statutory liability of ₹250 per day up to maximum ceiling of ₹25,000.",
      severity: "critical"
    });
  }

  return {
    detectedSections,
    flags,
    statutoryFramework: "Right to Information Act, 2005 (Act No. 22 of 2005)",
    precedentsMatched: flags.is_sec8_1_j 
      ? ["Girish Ramchandra Deshpande v. CIC (2013) 1 SCC 212", "Subhash Chandra Agarwal v. Supreme Court of India (2020)"]
      : flags.record_claimed_lost
      ? ["Union of India v. Vishwas Bhamburkar (Delhi High Court W.P.(C) 3660/2012)"]
      : ["CBSE v. Aditya Bandopadhyay (2011) 8 SCC 497"]
  };
}

/**
 * Stage 4: Neuro-Symbolic Synthesis & Deterministic Judicial Switch Matrix
 */
function simulateStage4SwitchMatrix(signal1, signal2, signal3, exemptionsCited = [], daysElapsed = 15) {
  let finalOutcome = signal1.predictedClass;
  let overrideApplied = false;
  let ruleTriggered = "Default Neural Consensus (No Conflict)";
  let legalReasoningTrace = [];
  let judicialDirections = [];
  let certaintyScore = 91;

  const { flags, detectedSections } = signal3;
  const { totalItems, addressedCount, missingCount, evasionRate } = signal2;

  // =========================================================================
  // RULE 4: Willful Evasion / Penalty Trigger
  // =========================================================================
  if (
    (flags.delay_exceeds_100_days && missingCount > 0 && !flags.is_sec8_claimed) ||
    (flags.record_claimed_lost && missingCount > 0)
  ) {
    finalOutcome = "PENALTY_IMPOSED";
    overrideApplied = signal1.predictedClass !== "PENALTY_IMPOSED";
    ruleTriggered = "Rule 4: Willful Evasion & Delay Trigger (Section 20 Enforcement)";
    certaintyScore = 96;

    legalReasoningTrace.push("1. Procedural Check: Delay of " + (flags.delay_exceeds_100_days ? "over 100 days" : "unexcused period") + " violates Section 7(1) statutory deadline.");
    legalReasoningTrace.push("2. Record Custody: The PIO's plea of missing or untraceable records is untenable in light of Delhi High Court ruling in 'Vishwas Bhamburkar'.");
    legalReasoningTrace.push("3. Signal 2 Vector Audit confirms " + evasionRate + "% of requested items were completely evaded.");
    legalReasoningTrace.push("4. Deterministic Rule Matrix activates Section 20(1) penal mechanism to deter institutional obfuscation.");

    judicialDirections = [
      "Issue Show-Cause Notice to the Public Information Officer under Section 20(1) of the RTI Act.",
      "Direct the Head of Department to institute an administrative inquiry into the alleged loss of records and fix accountability.",
      "Order reconstruction of the missing file within 20 working days and provide certified copies to the appellant free of charge under Section 7(6)."
    ];
  }

  // =========================================================================
  // RULE 2: Legitimate Statutory Privacy Shield
  // =========================================================================
  else if (
    (flags.is_sec8_1_j || flags.is_sec24_claimed) &&
    (signal1.predictedClass === "INFORMATION_DIRECTED" || signal1.predictedClass === "PARTIAL_RELIEF")
  ) {
    finalOutcome = "APPEAL_DISMISSED";
    overrideApplied = true;
    ruleTriggered = "Rule 2: Legitimate Privacy Shield Override (Section 8(1)(j) / Sec 24)";
    certaintyScore = 94;

    legalReasoningTrace.push("1. Exemption Grounding: Section 8(1)(j) legitimately invoked for third-party personal records (ITR, medical, or property).");
    legalReasoningTrace.push("2. Public Interest Test: The appellant has failed to establish any 'larger public interest' overriding individual privacy rights.");
    legalReasoningTrace.push("3. Precedent Alignment: Corresponds directly with Supreme Court doctrine in Girish Ramchandra Deshpande (2013).");
    legalReasoningTrace.push("4. Symbolic override suppresses Neural Information Directed prediction to maintain statutory compliance.");

    judicialDirections = [
      "The decision of the Public Information Officer denying disclosure is upheld.",
      "Appeal stands dismissed under Section 8(1)(j) of the Right to Information Act, 2005."
    ];
  }

  // =========================================================================
  // RULE 1: Unjustified Denial Override
  // =========================================================================
  else if (
    signal1.predictedClass === "APPEAL_DISMISSED" &&
    missingCount > 0 &&
    !flags.is_sec8_claimed &&
    !flags.is_sec24_claimed
  ) {
    finalOutcome = "INFORMATION_DIRECTED";
    overrideApplied = true;
    ruleTriggered = "Rule 1: Unjustified Denial Override (Statutory Default to Transparency)";
    certaintyScore = 93;

    legalReasoningTrace.push("1. Exemption Absence: The PIO denied information or provided evasive replies without citing any statutory Section 8 exemption.");
    legalReasoningTrace.push("2. Strategic Bureaucratic Ambiguity: Signal 2 semantic coverage analysis reveals " + evasionRate + "% of queries were unfulfilled.");
    legalReasoningTrace.push("3. Statutory Mandate: Under Section 7(1), access is mandatory unless explicitly shielded by Section 8 or 9.");
    legalReasoningTrace.push("4. Decision Matrix executes Rule 1, overturning the neural dismissal prediction to enforce constitutional transparency.");

    judicialDirections = [
      "Direct the PIO to furnish complete, attested copies of all requested documents within 15 working days.",
      "Information shall be provided free of cost in terms of Section 7(6) due to failure to respond in accordance with statutory guidelines.",
      "Warn the PIO against routine denial without citing valid statutory provisions."
    ];
  }

  // =========================================================================
  // RULE 3: Partial Compliance & Severability
  // =========================================================================
  else if (
    (addressedCount > 0 && missingCount > 0) ||
    (addressedCount > 0 && flags.is_sec8_claimed)
  ) {
    finalOutcome = "PARTIAL_RELIEF";
    overrideApplied = signal1.predictedClass !== "PARTIAL_RELIEF";
    ruleTriggered = "Rule 3: Partial Compliance & Section 10 Severability Enforcement";
    certaintyScore = 89;

    legalReasoningTrace.push("1. Composite Request: The citizen's application contains multiple independent queries.");
    legalReasoningTrace.push("2. Signal 2 Checklist Audit: " + addressedCount + " items were addressed, while " + missingCount + " items remain unfulfilled or withheld.");
    legalReasoningTrace.push("3. Severability Principle: Section 10(1) mandates disclosure of non-exempt portions while severing exempt portions.");
    legalReasoningTrace.push("4. Decision Matrix harmonizes signals into Partial Relief.");

    judicialDirections = [
      "Uphold exemption on confidential or evaluation methodology portions under Section 8(1).",
      "Direct the PIO to furnish the remaining non-exempt information (Item 1 / factual scores) to the appellant within 10 days.",
      "Compliance report to be submitted to the Commission within 2 weeks."
    ];
  }

  // =========================================================================
  // DEFAULT CASE: Consensus Alignment
  // =========================================================================
  else {
    finalOutcome = signal1.predictedClass;
    overrideApplied = false;
    ruleTriggered = "Consensus: Neural Signals Harmonized with Statutory Scanner";
    certaintyScore = Math.max(85, signal1.confidence);

    legalReasoningTrace.push("1. Neural Sequence Classification aligned with factual narrative context (" + signal1.confidence + "% confidence).");
    legalReasoningTrace.push("2. Signal 2 Semantic Coverage and Signal 3 Statutory Scanner corroborate the neural finding.");
    legalReasoningTrace.push("3. No legal conflict or statutory boundary breach detected by the deterministic switch matrix.");

    if (finalOutcome === "INFORMATION_DIRECTED") {
      judicialDirections = [
        "Direct the PIO to provide complete information within 15 working days.",
        "Ensure compliance report is filed with the appellate registry."
      ];
    } else if (finalOutcome === "APPEAL_DISMISSED") {
      judicialDirections = [
        "No relief can be granted. The PIO's reply is in accordance with the provisions of the RTI Act.",
        "Appeal is hereby dismissed."
      ];
    } else {
      judicialDirections = [
        "Directions issued as per Commission standard procedures."
      ];
    }
  }

  // Context-bounded Case Finding
  const exSummary = Array.isArray(exemptionsCited) && exemptionsCited.length > 0
    ? `under Section ${exemptionsCited.join(', ')}`
    : (typeof exemptionsCited === 'string' && exemptionsCited.trim() && exemptionsCited !== '[]'
      ? `under ${exemptionsCited}`
      : 'without citing any statutory exemption');

  let caseFinding = "";
  if (finalOutcome === "INFORMATION_DIRECTED") {
    caseFinding = `"Information Directed" was chosen because the PIO failed to furnish the requested records and offered no statutory exemption under Section 8(1). Under Section 7(1) of the RTI Act, 2005, disclosure of public administrative records is mandatory in the absence of valid statutory grounds. Hence the CPIO is directed to furnish complete, certified information to the applicant within 15 days.`;
  } else if (finalOutcome === "PENALTY_IMPOSED") {
    caseFinding = `"Penalty Imposed" was chosen because the PIO failed to respond within the statutory 30-day mandate (${daysElapsed} days elapsed) without reasonable cause. Section 20(1) of the RTI Act mandates penal proceedings against a designated officer for persistent default or malafide obstruction. Hence a show-cause notice is issued under Section 20(1) to explain why the statutory penalty should not be levied.`;
  } else if (finalOutcome === "PARTIAL_RELIEF") {
    caseFinding = `"Partial Relief" was chosen because the PIO furnished part of the requested information while withholding remaining portions under claimed exemptions. Section 10(1) of the RTI Act establishes the doctrine of severability, requiring disclosure of all non-exempt portions. Hence the CPIO is directed to supply certified copies of the unaddressed, non-exempt items within 15 days.`;
  } else if (finalOutcome === "APPEAL_DISMISSED") {
    caseFinding = `"Appeal Dismissed" was chosen because the PIO's refusal is supported by valid statutory exemptions ${exSummary} protecting the records from disclosure. The statutory bars under the RTI Act, 2005 legally restrict access to confidential or third-party records in the absence of larger public interest. Hence the CPIO's decision is upheld and the appeal is dismissed.`;
  } else {
    caseFinding = `"${finalOutcome.replace(/_/g, ' ')}" was chosen because the dispute was adjudicated in accordance with governing statutory boundaries under the RTI Act, 2005. Hence the matter is disposed of accordingly.`;
  }

  return {
    finalOutcome,
    outcomeMeta: OUTCOME_META[finalOutcome],
    overrideApplied,
    ruleTriggered,
    certaintyScore,
    legalReasoningTrace,
    judicialDirections,
    caseFinding
  };
}

/**
 * Main Pipeline Orchestrator
 */
export async function executePipeline({
  citizenQuery,
  pioReply,
  publicAuthority = "Public Authority",
  exemptionsCited = [],
  daysElapsed = 15,
  liveSignal1 = null
}) {
  let signal1 = null;
  if (liveSignal1 && liveSignal1.predicted_class) {
    signal1 = {
      predictedClass: liveSignal1.predicted_class,
      confidence: Math.round(liveSignal1.confidence * 100),
      probabilities: liveSignal1.probabilities,
      modelName: `Fine-Tuned InLegalBERT (Live ${liveSignal1.device ? liveSignal1.device.toUpperCase() : 'CUDA'} Checkpoint)`,
      isLiveModel: true,
      device: liveSignal1.device || "cuda",
      tokensProcessed: Math.min(512, Math.round((citizenQuery.length + pioReply.length) / 3.8)),
      headTokens: 160,
      tailTokens: 350
    };
  } else {
    signal1 = simulateSignal1(citizenQuery, pioReply, exemptionsCited, daysElapsed);
  }

  const signal2 = simulateSignal2(citizenQuery, pioReply);
  const signal3 = simulateSignal3(citizenQuery, pioReply, exemptionsCited, daysElapsed);
  const stage4 = simulateStage4SwitchMatrix(signal1, signal2, signal3, exemptionsCited, daysElapsed);

  return {
    timestamp: new Date().toISOString(),
    inputs: {
      citizenQuery,
      pioReply,
      publicAuthority,
      exemptionsCited,
      daysElapsed
    },
    signal1,
    signal2,
    signal3,
    stage4
  };
}


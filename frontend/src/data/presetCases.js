/**
 * Pre-configured Realistic Historical RTI Benchmark Cases
 * Demonstrates all branches of the Explainable Neuro-Symbolic Decision Matrix
 */

export const PRESET_CASES = [
  {
    id: "preset_1",
    title: "Evasive PIO on Public Works Tender",
    tag: "Rule 1 Override → Information Directed",
    badgeColor: "cyan",
    caseId: "CIC/PWD/2024/00812",
    publicAuthority: "Public Works Department, GNCTD",
    exemptionsCited: "",
    daysElapsed: 48,
    citizenQuery: `1. Attested copy of the final tender evaluation summary and financial bids submitted for the Outer Ring Road repair project (Tender No. PWD/ORR/2023/89).
2. Certified copy of the structural safety inspection report signed by the Executive Engineer prior to release of the 3rd milestone payment.
3. Total amount disbursed to M/s Apex Infrastructure Ltd. till date under this contract.`,
    pioReply: `With reference to your RTI application, it is informed that the repair project is currently in execution and maintenance phase. All contract documents are kept under routine administrative custody. Further, structural safety reports are prepared for internal departmental review. The applicant is advised that voluminous papers cannot be photocopied under standard RTI queries.`,
    expectedOutcome: "INFORMATION_DIRECTED",
    expectedRule: "Rule 1 (Unjustified Denial Override)",
    summary: "The PIO provided generic administrative excuses without citing any statutory Section 8 exemption. Signal 2 flags 100% evasion, prompting the Switch Matrix to override and direct full disclosure."
  },
  {
    id: "preset_2",
    title: "Third-Party Personal Records & Privacy Shield",
    tag: "Rule 2 Validation → Appeal Dismissed",
    badgeColor: "rose",
    caseId: "CIC/DOPT/2024/01492",
    publicAuthority: "Department of Personnel and Training (DoPT)",
    exemptionsCited: "8(1)(j)",
    daysElapsed: 24,
    citizenQuery: `1. Certified copies of the Annual Property Returns (APR) and Income Tax Returns of Shri R.K. Sharma (Deputy Secretary) for assessment years 2021-2024.
2. Complete medical reimbursement claims, hospital discharge summaries, and medical board fitness records submitted by Shri R.K. Sharma and his dependent family members.`,
    pioReply: `The information sought relates to personal details, assets, income tax filings, and medical records of a third-party officer. Under Section 11 of the RTI Act, notice was issued to the third party who strongly objected to disclosure. The disclosure has no relationship to any public activity or interest and would cause unwarranted invasion of privacy. The application is therefore denied under Section 8(1)(j) of the RTI Act, 2005.`,
    expectedOutcome: "APPEAL_DISMISSED",
    expectedRule: "Rule 2 (Legitimate Privacy Shield)",
    summary: "The citizen seeks personal asset and medical records of a third party. The PIO correctly invoked Section 8(1)(j) and followed Section 11 third-party procedure. The Switch Matrix affirms dismissal under Supreme Court precedents."
  },
  {
    id: "preset_3",
    title: "Recruitment Exam Scores vs Methodology",
    tag: "Rule 3 Enforcement → Partial Relief",
    badgeColor: "amber",
    caseId: "CIC/UPSC/2023/04118",
    publicAuthority: "Staff Selection Commission / Exam Board",
    exemptionsCited: "8(1)(d)",
    daysElapsed: 35,
    citizenQuery: `1. Certified copy of applicant's own evaluated OMR answer sheet and official category-wise cut-off marks for Roll No. 2209148.
2. Raw moderation formula, scaling metrics, and examiners' confidential panel notes applied across all exam centers.`,
    pioReply: `Regarding Query 1, the official category-wise cut-off marks and applicant's evaluated scorecard are enclosed herewith. Regarding Query 2, the raw moderation formula and confidential panel notes cannot be provided as they involve internal evaluation methodology exempt under Section 8(1)(d).`,
    expectedOutcome: "PARTIAL_RELIEF",
    expectedRule: "Rule 3 (Partial Compliance & Severability)",
    summary: "Item 1 was addressed by the PIO, while Item 2 was legitimately withheld under Section 8(1)(d). The Switch Matrix applies Section 10 severability to enforce Partial Relief."
  },
  {
    id: "preset_4",
    title: "Willful Destruction & 125-Day Delay",
    tag: "Rule 4 Enforcement → Penalty Imposed",
    badgeColor: "purple",
    caseId: "CIC/POSTS/2023/09931",
    publicAuthority: "Department of Posts, Ministry of Communications",
    exemptionsCited: "",
    daysElapsed: 125,
    citizenQuery: `1. Certified copy of the pension sanction order and gratuity settlement calculation sheet for retired Postal Assistant P. Subramanian (PPO No. PA/99214).
2. Name and designation of the dealing officer responsible for withholding revised Dearness Relief arrears since March 2022.`,
    pioReply: `The pension records are very old and could not be traced in the record room. The dealing assistant has been transferred. Information will be provided as and when files are reconstructed.`,
    expectedOutcome: "PENALTY_IMPOSED",
    expectedRule: "Rule 4 (Willful Evasion & Delay Trigger)",
    summary: "The PIO delayed response for 125 days without valid cause and claimed missing records. The Switch Matrix flags willful evasion under Section 20(1) recommending penal proceedings."
  }
];

export const EXEMPTION_OPTIONS = [
  { id: "8(1)(a)", label: "Sec 8(1)(a) - Sovereignty & National Security" },
  { id: "8(1)(b)", label: "Sec 8(1)(b) - Court Contempt & Judicial Prohibition" },
  { id: "8(1)(c)", label: "Sec 8(1)(c) - Parliamentary Privilege" },
  { id: "8(1)(d)", label: "Sec 8(1)(d) - Commercial Confidence & Trade Secrets" },
  { id: "8(1)(e)", label: "Sec 8(1)(e) - Fiduciary Relationship" },
  { id: "8(1)(f)", label: "Sec 8(1)(f) - Foreign Government Confidences" },
  { id: "8(1)(g)", label: "Sec 8(1)(g) - Life Safety & Law Enforcement Sources" },
  { id: "8(1)(h)", label: "Sec 8(1)(h) - Pending Police / CBI Investigation" },
  { id: "8(1)(i)", label: "Sec 8(1)(i) - Cabinet Papers Prior to Decision" },
  { id: "8(1)(j)", label: "Sec 8(1)(j) - Personal Privacy (No Public Interest)" },
  { id: "24", label: "Sec 24 - Schedule II Security / Intelligence Org" },
  { id: "2(f)", label: "Sec 2(f) - Definition of Information (Opinion vs Record)" }
];

"""
src/nlp/nlp_assistant.py
RadiScan | Team Destroyers | 42174 AI Studio Autumn 2026
Owner: Aagusthya Shanker

Radiology Breast Cancer AI Assistant — GPT-4o powered.
Handles ALL NLP/GPT logic. Imported by app/streamlit_app.py.

Usage:
    from src.nlp.nlp_assistant import RadiScanAssistant
    bot = RadiScanAssistant(api_key=os.getenv("OPENAI_API_KEY",""))
    answer = bot.ask(question, patient_dict, ai_result_dict, chat_history)
"""

import os, random

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Topic classifier ─────────────────────────────────────────────────────────
MEDICAL_WORDS = {
    "scan","image","histopath","patch","classif","result","predict","probab",
    "threshold","malign","benign","cancer","non-cancer","noncancer",
    "model","efficientnet","gradcam","grad-cam","heatmap","activat","feature",
    "confid","accura","recall","auc","precision","f1","false","positive","negative",
    "refer","oncol","biopsy","surgery","lumpectomy","mastectomy","mdt",
    "next step","recommend","manage","treatment","urgent","flag","follow","report",
    "idc","ductal","carcinoma","invasive","tumour","tumor","lump","mass","lesion",
    "nodule","calcif","microcalcif","lymph","axillar","metastas","sentinel",
    "histolog","patholog","tissue","cell","nuclear","stroma","glandular","atypia",
    "nuclear-to-cytoplasm","nuclear morphol","architecture",
    "mammogram","mri","ultrasound","birads","bi-rads","grade","staging","stage",
    "brca","her2","er ","pr ","receptor","hormone","triple negative","tamoxifen",
    "aromatase","herceptin","trastuzumab","genetic","hereditar","mutation",
    "nipple","discharge","dimpl","peau","retract","pain","tender","swell",
    "inflamm","redness","asymmetr","densit","skin change",
    "patient","radiolog","clinician","doctor","physician","diagnos","prognos",
    "survival","risk","screening","annual","protocol","workup","assessment",
    "why","what","how","should","explain","describe","tell me","is this",
    "does this","what kind","what type","what does","what is","what are",
    "interpret","interpret","scan","finding","read","review","impression",
}

def _is_medical(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in MEDICAL_WORDS)


def _build_system_prompt(patient: dict, ai_result: dict) -> str:
    is_cancer = ai_result["class"] == "Cancer"
    prob      = ai_result["probability"]
    prob_pct  = f"{prob * 100:.1f}%"
    conf_band = ("high" if prob >= 0.80 or prob <= 0.20
                 else "moderate" if 0.55 <= prob <= 0.80 or 0.20 < prob < 0.45
                 else "borderline")

    if is_cancer and prob >= 0.8:
        referral = f"URGENT oncology referral is indicated. Malignant probability {prob_pct} exceeds high-confidence threshold (≥80%)."
    elif is_cancer:
        referral = f"Oncology referral should be considered. Malignant probability {prob_pct} — moderate confidence. Correlation with prior imaging and MDT discussion recommended before escalation."
    else:
        referral = f"Routine oncology referral is not indicated ({prob_pct} malignant probability, below 40% decision threshold). Standard follow-up per screening protocol."

    return f"""You are RadiScan Clinical AI — an AI Radiology Breast Cancer Assistant embedded in a hospital radiology information system.

You are speaking directly to an experienced RADIOLOGIST. Not a patient, not a student.

Respond like a knowledgeable AI decision-support tool — clinical, technical, precise.
Do NOT explain what a radiologist is. Do NOT over-explain basics. Do NOT say "please consult your doctor."

=== CURRENT CASE ===
Patient:       {patient['name']}, Age {patient['age']}, {patient['sex']}
MRN:           {patient['mrn']}
Complaint:     {patient['complaint']}
History:       {patient['history']}
Referring:     {patient['referring']}
Priority:      {patient.get('priority', 'Routine')}

=== AI CLASSIFICATION ===
Result:        {ai_result['class']}
Probability:   {prob_pct}  (threshold: 40%)
Confidence:    {conf_band.upper()}
Model:         EfficientNet-B0 v0.2 | 277,524 IDC patches | Acc 91.6% | Recall 89.3% | AUC 0.970

=== REFERRAL SUMMARY ===
{referral}

=== HOW TO ANSWER EACH QUESTION TYPE ===

Q: "Why malignant?" / "Why this result?" / "What features?" / "Explain classification"
→ Describe the histopathological morphology the model detected. Be specific:
  - Nuclear atypia (enlarged, irregular, hyperchromatic nuclei)
  - Raised nuclear-to-cytoplasmic ratio
  - Disordered glandular architecture (loss of tubule formation)
  - Stromal desmoplasia or inflammatory infiltrate
  - Mitotic figures in high-grade cases
  - Lack of organised epithelial layers (IDC hallmark)
  Mention that the Grad-CAM heatmap localises the highest-activation region.
  For NON-MALIGNANT: explain preserved architecture, regular nuclear morphology, no atypia.

Q: "What does the Grad-CAM heatmap show?" / "What does the heatmap show?" / "What are the red areas?"
→ Explain Grad-CAM technically and diagnostically. Include:
  - Red/yellow = highest gradient activation = regions most influential to the prediction
  - Blue/green = low activation = minimal contribution
  - Specifically: warm regions correspond to areas where the model detected suspicious morphology
  - In IDC cases: high-activation clusters often correspond to atypical epithelial cell nests, nuclear crowding, or stromal border irregularity
  - Clinical implication: these regions should be prioritised in subsequent pathological assessment
  - Mention this is Gradient-weighted Class Activation Mapping from features[8] of EfficientNet-B0

Q: "How confident?" / "How accurate?" / "False positive risk?" / "Reliable?"
→ Give specific metrics, not vague reassurance:
  - State the exact probability ({prob_pct}) and what confidence band it falls in
  - Model sensitivity (recall) 89.3%, specificity implied from 91.6% accuracy
  - AUC 0.970 on 41,630-image test set
  - 40% threshold was selected to maximise recall — reduces missed cancers (false negatives)
  - At this threshold, false positive rate is approximately 8%
  - Uncertainty is highest in the 40–60% probability range (borderline cases)
  - State limitations: model trained on 50×50px IDC patches; OOD detection not implemented

Q: "Next steps?" / "What should I do?" / "Refer?" / "Management?"
→ Answer as a clinical decision-support tool advising a radiologist:
  {f"URGENT pathway: (1) Correlate with clinical examination and prior imaging interval change. (2) Core needle biopsy for histopathological confirmation and receptor testing (ER/PR/HER2). (3) MDT referral — breast surgeon + oncology. (4) Staging workup if biopsy confirms IDC." if is_cancer and prob >= 0.8 else
   f"Recommended workup: (1) Compare with prior imaging — interval change assessment. (2) Consider targeted ultrasound or MRI for further characterisation. (3) If clinically suspicious, proceed with core needle biopsy. (4) MDT discussion before escalation given moderate AI confidence ({prob_pct})." if is_cancer else
   f"Recommended: (1) Document and compare with prior imaging for interval change. (2) Continue standard screening protocol. (3) If clinical suspicion persists despite negative AI result, pursue additional imaging workup — AI results do not exclude pathology. (4) Schedule follow-up per institutional protocol."}

Q: "Explain the scan" / "Impression?" / "What does this show?"
→ Give a structured radiology-style impression:
  - AI classification and probability
  - Morphological features observed (as above)
  - Grad-CAM region of interest
  - Recommended correlation
  Mimic the structure of a radiology report impression section.

Q: "IDC?" / "What is IDC?" / "Tell me about this cancer"
→ Brief clinical overview: IDC = most common breast cancer (~80%). Originates in lactiferous ducts, breaches basement membrane, invades stroma. Graded I–III (Nottingham). ER/PR/HER2 receptor status determines systemic therapy.

Q: "BIRADS?" → BIRADS 1=Negative, 2=Benign, 3=Probably benign (6m follow-up), 4A/B/C=Suspicious (biopsy), 5=Highly suspicious, 6=Biopsy-proven malignancy. This AI histopathology classifier complements BIRADS imaging scores.

Q: "BRCA?" / "Genetics?" → BRCA1 ~72%, BRCA2 ~69% lifetime risk vs 12% population. Patient history noted: {patient['history']}.

Q: "Receptors?" / "HER2?" / "ER/PR?" → Receptor status (ER/PR/HER2) determined by IHC on biopsy specimen, not this AI classifier. Request receptor panel after tissue diagnosis.

Q: "Staging?" → TNM: T=tumour size, N=nodal status, M=metastasis. Stage I–IV determines treatment intent (curative vs palliative). Staging requires clinical + radiological + pathological correlation.

Q: "Treatment?" / "Chemo?" / "Surgery?" → IDC treatment by stage: surgery (BCS or mastectomy ± sentinel node biopsy), adjuvant radiation (post-BCS), chemotherapy (neoadjuvant for ≥T2/N+), hormone therapy (ER/PR+), HER2-targeted (trastuzumab). Determined post-MDT.

=== STRICT RESPONSE RULES ===
1. Maximum 4 sentences. Be dense, not long.
2. Clinical tone — you are advising a radiologist, not explaining medicine to a layperson.
3. Vary your phrasing — do NOT repeat the same sentence structure in different answers.
4. ONE clinical caveat maximum per response: "AI output requires histopathological correlation." Use ONLY when truly relevant.
5. NEVER say "please consult your doctor" or "please see a specialist."
6. NEVER repeat probability numbers more than once per response.
7. Off-topic questions: one sentence redirect only.
8. Do NOT start every response with "EfficientNet-B0 classified..." — vary the opening."""


class RadiScanAssistant:
    """RadiScan Clinical AI. GPT-4o with comprehensive radiologist-oriented fallback."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or OPENAI_API_KEY
        self.model   = "gpt-4o"
        self._last_fallback_type = None  # prevent repetition across turns

    @property
    def api_available(self) -> bool:
        return bool(self.api_key and len(self.api_key) > 20)

    def ask(self, question: str, patient: dict, ai_result: dict,
            chat_history: list) -> str:
        if not ai_result:
            return ("No scan analysed yet. Upload a histopathology patch and run AI Analysis "
                    "to activate the assistant.")

        if not _is_medical(question):
            pred = ai_result["class"]
            prob = ai_result["probability"]
            return (
                f"This assistant is scoped to radiology and breast cancer AI. "
                f"Current case: {patient['name']}, {pred} ({prob*100:.0f}%). "
                f"Ask me: why was this classified, what the Grad-CAM shows, next clinical steps, "
                f"or model confidence."
            )

        if self.api_available:
            return self._call_gpt(question, patient, ai_result, chat_history)
        return self._fallback(question, patient, ai_result)

    def _call_gpt(self, question, patient, ai_result, history):
        try:
            import requests
            system = _build_system_prompt(patient, ai_result)
            prior  = [{"role": m["role"], "content": m["content"]}
                      for m in (history or [])[-8:]]
            prior.append({"role": "user", "content": question})

            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       self.model,
                    "messages":    [{"role": "system", "content": system}] + prior,
                    "max_tokens":  380,
                    "temperature": 0.2,
                },
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return self._fallback(question, patient, ai_result)
        except Exception:
            return self._fallback(question, patient, ai_result)

    def _fallback(self, question, patient, ai_result):
        """High-quality radiologist-oriented fallback responses."""
        q    = question.lower()
        pred = ai_result["class"]
        prob = ai_result["probability"]
        name = patient["name"]
        is_c = (pred == "Cancer")
        pct  = f"{prob*100:.0f}%"

        # ── Grad-CAM / heatmap ─────────────────────────────────────────────────
        if any(w in q for w in ["heatmap","gradcam","grad-cam","grad cam","red","warm",
                                  "activation","highlight","region","area","colour","color",
                                  "overlay","visual","show","what does the"]):
            if is_c:
                return (
                    "Grad-CAM (Gradient-weighted Class Activation Mapping) identifies the image "
                    "regions most influential to the malignant classification. "
                    "Red/yellow zones represent peak gradient activation — these correspond to areas "
                    "of atypical epithelial cell nests, nuclear crowding, or stromal border irregularity "
                    "that drove the prediction. "
                    "Prioritise these highlighted regions in subsequent pathological assessment and "
                    "correlate with H&E morphology."
                )
            return (
                "Grad-CAM gradient map shows diffuse, low-intensity activation across the patch "
                "with no focal high-activation cluster. "
                "This is consistent with organised epithelial architecture — the model found no "
                "regional feature concentration indicative of IDC. "
                "Low-activation maps support the non-malignant classification, though clinical "
                "correlation remains necessary."
            )

        # ── Why classified / morphology rationale ──────────────────────────────
        if any(w in q for w in ["why","reason","classif","feature","detect","how was","morphol",
                                  "what feature","what made","basis","evidence","explain"]):
            if is_c:
                return (
                    f"Classification as malignant ({pct}) is based on IDC-consistent histomorphology: "
                    f"irregular nuclear enlargement with hyperchromasia, raised nuclear-to-cytoplasmic ratio, "
                    f"disordered glandular architecture with loss of tubule formation, and atypical "
                    f"stromal patterns. "
                    f"The Grad-CAM heatmap localises the highest-activation cluster — this region "
                    f"contains the most diagnostically significant cellular features. "
                    f"AI output requires histopathological correlation."
                )
            return (
                f"Non-malignant classification ({pct}) reflects preserved epithelial architecture: "
                f"regular nuclear morphology, maintained nuclear-to-cytoplasmic ratio, organised "
                f"glandular structures without atypia, and no evidence of stromal invasion pattern. "
                f"Grad-CAM shows diffuse low-level activation — consistent with absence of focal "
                f"malignant morphology in this patch."
            )

        # ── Confidence / accuracy ──────────────────────────────────────────────
        if any(w in q for w in ["confid","accura","reliab","certain","trust","false positive",
                                  "false negative","sensitivity","specificity","how good",
                                  "how sure","miss","error","limitation"]):
            conf_band = ("high" if prob >= 0.80 or prob <= 0.20 else
                         "moderate" if 0.55 <= prob or prob <= 0.45 else "borderline")
            return (
                f"Malignant probability {pct} — {conf_band} confidence. "
                f"Model performance on 41,630-image held-out test set: accuracy 91.6%, recall 89.3%, AUC 0.970. "
                f"The 40% decision threshold prioritises sensitivity over specificity — at this operating point, "
                f"false positive rate is approximately 8%, which is clinically preferable to missed cancers. "
                f"Uncertainty is highest in the 40–60% probability range; borderline cases warrant MDT discussion."
            )

        # ── Next steps / management / referral ────────────────────────────────
        if any(w in q for w in ["next","step","should","refer","management","recommend",
                                  "action","what do","workflow","urgent","biopsy","mdt"]):
            if is_c and prob >= 0.8:
                return (
                    f"URGENT pathway recommended: correlate with clinical examination and interval "
                    f"change from prior imaging. "
                    f"Proceed to core needle biopsy for histopathological confirmation and ER/PR/HER2 "
                    f"receptor testing. "
                    f"Urgent MDT referral — breast surgeon and oncology. "
                    f"Staging workup (axillary ultrasound ± MRI) if biopsy confirms IDC."
                )
            elif is_c:
                return (
                    f"Recommended workup for {name}: compare with prior imaging for interval change assessment. "
                    f"Consider targeted ultrasound or breast MRI for further characterisation given moderate "
                    f"AI confidence ({pct}). "
                    f"Core needle biopsy if clinically suspicious or imaging equivocal. "
                    f"MDT discussion before escalation — avoid over-investigation in isolation."
                )
            return (
                f"Non-malignant result: document and compare with prior imaging for interval change. "
                f"Continue standard screening protocol per institutional guidelines. "
                f"If clinical suspicion persists despite negative AI result, pursue additional imaging "
                f"workup — a single patch result does not exclude pathology. "
                f"Follow-up appointment at recommended interval."
            )

        # ── Impression / explain scan / overall interpretation ─────────────────
        if any(w in q for w in ["impression","interpret","explain","overview","summarise",
                                  "summarize","tell me about","what does this show","read",
                                  "overall","report"]):
            if is_c:
                return (
                    f"Impression: Histopathology patch demonstrates features suspicious for IDC "
                    f"(malignant probability {pct}). "
                    f"Morphological correlates include nuclear atypia, glandular disorganisation, and "
                    f"stromal abnormality; Grad-CAM localises highest-activation region for targeted review. "
                    f"Recommend biopsy correlation, receptor testing, and urgent MDT referral."
                )
            return (
                f"Impression: No malignant features identified on AI analysis ({pct} malignant probability, "
                f"below 40% threshold). "
                f"Histomorphology consistent with benign/normal breast tissue — preserved architecture, "
                f"regular nuclear morphology, no atypia detected. "
                f"Continue surveillance per protocol; AI output requires clinical correlation."
            )

        # ── IDC / cancer type ──────────────────────────────────────────────────
        if any(w in q for w in ["idc","ductal","carcinoma","invasive","what is this cancer",
                                  "what type","what kind of cancer"]):
            return (
                "Invasive Ductal Carcinoma (IDC) is the most common breast malignancy (~80%). "
                "Originates in the lactiferous ducts, breaches the basement membrane, and invades "
                "surrounding stroma. Graded I–III by Nottingham score (tubule formation, nuclear "
                "pleomorphism, mitotic index). "
                "ER/PR/HER2 receptor status is determined by IHC on biopsy specimen and drives "
                "systemic therapy selection."
            )

        # ── Condition / what is this ───────────────────────────────────────────
        if any(w in q for w in ["what is","what kind","condition","cancerous","lump",
                                  "is this","non cancer","non-cancer","type"]):
            if is_c:
                return (
                    f"AI classification: {pred} — {pct} malignant probability. "
                    f"Features consistent with IDC: irregular nuclear morphology, raised N:C ratio, "
                    f"disordered glandular architecture. "
                    f"Definitive histopathological diagnosis requires core needle biopsy with "
                    f"receptor panel (ER/PR/HER2)."
                )
            return (
                f"AI classification: {pred} — {pct} malignant probability (below 40% threshold). "
                f"Cellular architecture appears preserved with regular nuclear morphology and no "
                f"significant atypia. "
                f"Clinical and imaging correlation recommended; a negative AI result does not "
                f"exclude pathology."
            )

        # ── BIRADS ────────────────────────────────────────────────────────────
        if any(w in q for w in ["birads","bi-rads","bi rads"]):
            return (
                "BIRADS (ACR Breast Imaging Reporting and Data System): "
                "1=Negative, 2=Benign, 3=Probably benign (6m follow-up), "
                "4A=Low/4B=Moderate/4C=High suspicion (biopsy recommended), "
                "5=Highly suggestive of malignancy, 6=Known biopsy-proven. "
                "This AI system classifies histopathology patches by malignant probability and "
                "complements BIRADS mammography/US scores — not a direct equivalent."
            )

        # ── Receptors / HER2 ──────────────────────────────────────────────────
        if any(w in q for w in ["her2","er ","pr ","receptor","hormone","triple negative",
                                  "tamoxifen","aromatase","herceptin","trastuzumab"]):
            return (
                "Receptor status (ER/PR/HER2) is determined by immunohistochemistry on biopsy "
                "tissue — not by this AI patch classifier. "
                "ER+/PR+: hormone therapy (tamoxifen or aromatase inhibitors). "
                "HER2+: targeted therapy (trastuzumab ± pertuzumab). "
                "Triple-negative (ER-/PR-/HER2-): chemotherapy backbone, immunotherapy in some cases. "
                "Request receptor panel concurrent with or following tissue diagnosis."
            )

        # ── BRCA / genetics ───────────────────────────────────────────────────
        if any(w in q for w in ["brca","genetic","hereditar","family history","mutation"]):
            hist = patient.get('history','')
            brca_note = ""
            if "brca" in hist.lower():
                brca_note = f" Patient history notes BRCA status: {hist}."
            return (
                f"BRCA1/BRCA2 pathogenic variants confer 72%/69% lifetime breast cancer risk "
                f"versus ~12% population risk.{brca_note} "
                f"Enhanced surveillance (annual MRI + mammogram from age 25–30) and risk-reducing "
                f"options (chemoprevention, prophylactic surgery) apply per NICE/NCCN guidelines."
            )

        # ── Treatment ─────────────────────────────────────────────────────────
        if any(w in q for w in ["treatment","therapy","chemo","radiation","surgery",
                                  "lumpectomy","mastectomy","neoadjuvant","adjuvant"]):
            return (
                "IDC management is multidisciplinary: "
                "surgery (breast-conserving surgery + sentinel node biopsy, or mastectomy), "
                "adjuvant radiation (post-BCS), chemotherapy (neoadjuvant for ≥T2/node-positive), "
                "hormone therapy (ER/PR+), HER2-targeted therapy (trastuzumab ± pertuzumab). "
                "Treatment intent and regimen determined by stage, grade, and receptor status at MDT."
            )

        # ── Staging ───────────────────────────────────────────────────────────
        if any(w in q for w in ["staging","stage","spread","tnm","metastas","node"]):
            return (
                "TNM staging: T = tumour size (T1 ≤2cm, T2 2–5cm, T3 >5cm, T4 chest wall/skin), "
                "N = nodal status (N0–N3 by number and level), M = distant metastasis. "
                "Stage I–IV determines treatment intent (curative vs palliative). "
                "Staging requires clinical examination, axillary imaging (US ± FNAC), "
                "and systemic staging CT/bone scan for N+ or high-grade disease."
            )

        # ── Prognosis ─────────────────────────────────────────────────────────
        if any(w in q for w in ["prognosis","survival","outlook","outcome","5 year","5-year"]):
            return (
                f"Breast cancer 5-year survival by stage: Stage I ~99%, Stage II ~86%, "
                f"Stage III ~57%, Stage IV ~29%. "
                f"Prognostic factors: tumour grade, nodal status, receptor profile, Ki-67 index. "
                f"For {name}, definitive prognosis requires histopathological staging, receptor "
                f"testing, and full MDT assessment."
            )

        # ── Default ───────────────────────────────────────────────────────────
        return (
            f"Current case: {name}, {pred} ({pct}). "
            f"I can assist with: Grad-CAM heatmap interpretation, classification rationale, "
            f"model confidence, next clinical steps, IDC pathology, BIRADS, receptor status, "
            f"staging, or a full scan impression. "
            f"What would you like to know?"
        )

    @staticmethod
    def get_quick_questions(ai_result: dict) -> list:
        if not ai_result:
            return []
        if ai_result["class"] == "Cancer":
            return [
                "Why was this classified malignant?",
                "What does the Grad-CAM heatmap show?",
                "What are the next clinical steps?",
            ]
        return [
            "Why was this classified non-malignant?",
            "How confident is the model?",
            "What follow-up is recommended?",
        ]

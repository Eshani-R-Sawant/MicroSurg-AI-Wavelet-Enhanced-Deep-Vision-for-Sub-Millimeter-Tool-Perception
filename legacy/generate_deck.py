from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

# ---------- Palette ----------
NAVY = RGBColor(0, 0, 77)       # Primary #00004D
STEEL = RGBColor(62, 92, 138)   # Secondary accent
GOLD = RGBColor(180, 121, 26)   # Warm accent
ICE = RGBColor(231, 236, 245)   # Light card fill
ICE2 = RGBColor(243, 246, 251)
GOLD_BG = RGBColor(251, 240, 223)
WHITE = RGBColor(255, 255, 255)
TEXT = RGBColor(27, 27, 47)
MUTED = RGBColor(91, 100, 114)

FONT = "Times New Roman"
MONO = "Courier New"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
blank_layout = prs.slide_layouts[6]

# ---------- Helpers ----------
def add_chrome(slide, title_text, kicker=""):
    if kicker:
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.333), Inches(0.4))
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = kicker.upper()
        p.font.name = FONT
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = GOLD

    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(0.6 if kicker else 0.4), Inches(12.333), Inches(0.8))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.name = FONT
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = NAVY

    footer_y = Inches(7.0)
    footer_h = Inches(0.5)
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), footer_y, SLIDE_W, footer_h)
    rect.fill.solid()
    rect.fill.fore_color.rgb = NAVY
    rect.line.fill.background()

    tf = rect.text_frame
    p = tf.paragraphs[0]
    p.text = "INDIAN INSTITUTE OF TECHNOLOGY JODHPUR    •    DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING"
    p.font.name = FONT
    p.font.size = Pt(10)
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

def add_card(slide, left, top, width, height, title, items, bg_color=ICE, title_color=NAVY):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.fill.background()

    tx_box = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.2), width - Inches(0.4), Inches(0.6))
    tf = tx_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.name = FONT
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = title_color

    tx_box2 = slide.shapes.add_textbox(left + Inches(0.2), top + Inches(0.8), width - Inches(0.4), height - Inches(1.0))
    tf2 = tx_box2.text_frame
    tf2.word_wrap = True
    for idx, item in enumerate(items):
        p = tf2.add_paragraph() if idx > 0 else tf2.paragraphs[0]
        p.text = f"•  {item}"
        p.font.name = FONT
        p.font.size = Pt(11)
        p.font.color.rgb = TEXT
        p.space_after = Pt(6)

def add_table_slide(slide, rows_data, left, top, width, height, col_widths=None):
    rows = len(rows_data)
    cols = len(rows_data[0])
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    if col_widths:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = w

    for r_idx, row in enumerate(rows_data):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.name = FONT
                p.font.size = Pt(10)
                if r_idx == 0:
                    p.font.bold = True
                    p.font.color.rgb = WHITE
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = NAVY
                else:
                    p.font.color.rgb = TEXT
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = ICE2 if r_idx % 2 == 1 else WHITE

# ================= SLIDE 1 =================
slide = prs.slides.add_slide(blank_layout)
txBox = slide.shapes.add_textbox(Inches(1), Inches(1.2), Inches(11.333), Inches(4.5))
tf = txBox.text_frame
tf.word_wrap = True

p1 = tf.paragraphs[0]
p1.text = "INDIAN INSTITUTE OF TECHNOLOGY JODHPUR\nDEPARTMENT OF COMPUTER SCIENCE & ENGINEERING\n"
p1.font.name = FONT
p1.font.size = Pt(16)
p1.font.color.rgb = NAVY
p1.alignment = PP_ALIGN.CENTER

p2 = tf.add_paragraph()
p2.text = "CAD DETECTION PIPELINE v2\n"
p2.font.name = FONT
p2.font.size = Pt(36)
p2.font.bold = True
p2.font.color.rgb = NAVY
p2.alignment = PP_ALIGN.CENTER

p3 = tf.add_paragraph()
p3.text = "Multimodal Fusion of Vision Foundation Representations, Deterministic Anatomical Geometry\nand Attention-MIL for Coronary Artery Disease Screening from Chest Radiographs"
p3.font.name = FONT
p3.font.size = Pt(14)
p3.font.italic = True
p3.font.color.rgb = MUTED
p3.alignment = PP_ALIGN.CENTER

# ================= SLIDE 2 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Clinical Objective & Core Components", "Project Overview")
components = [
    ("1. CheXWorld ViT-B/16", ["768-dim self-supervised visual embedding pretrained on CXR imagery."]),
    ("2. Mask-CNN Encoder", ["128-dim deep spatial anatomical representation over organ masks."]),
    ("3. CheXStruct Geometry", ["Row-scan CTR and Aortic Knob Enlargement Ratio."]),
    ("4. Attention MIL Head", ["Aggregates multi-image encounters (Ilse et al., ICML 2018)."])
]
for i, (title, descs) in enumerate(components):
    add_card(slide, Inches(0.5) + i * Inches(3.1), Inches(2.0), Inches(2.9), Inches(4.5), title, descs, bg_color=ICE if i % 2 == 0 else GOLD_BG)

# ================= SLIDE 3 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Dataset Specification & Cohort Summary", "Cohort Summary")
stats = [("7,700", "Chest Radiographs"), ("3,808", "Unique Patients"), ("3.3 : 1", "CAD Class Ratio")]
for i, (num, desc) in enumerate(stats):
    add_card(slide, Inches(0.5), Inches(1.5 + i * 1.7), Inches(3.5), Inches(1.4), num, [desc], bg_color=ICE)
rows3 = [
    ["Resolution / Format", "Purpose"],
    ["512 × 512", "CXAS multi-organ segmentation input"],
    ["224 × 224, norm [-1, 1]", "CheXWorld ViT-B/16 foundation model input"]
]
add_table_slide(slide, rows3, Inches(4.5), Inches(1.5), Inches(8.3), Inches(2.0))

# ================= SLIDE 4 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "End-to-End Technical Workflow", "Pipeline Architecture")
steps = [
    "Raw CXR Image (512x512 / 224x224)",
    "Segmentation (CXAS) & Foundation Encoding (CheXWorld ViT-B/16)",
    "CheXStruct Geometry + Mask-CNN Spatial Encoder + Global Embedding",
    "5-Fold Stratified Group K-Fold Split (Zero Data Leakage)",
    "Multi-Modal LP-FT Late Fusion Classifier (Focal Loss, γ=2.0)",
    "Temporal Stratified Validation & Encounter-Level Attention MIL"
]
for i, st in enumerate(steps):
    add_card(slide, Inches(1.5), Inches(1.3 + i * 0.9), Inches(10.333), Inches(0.75), f"Stage {i+1}", [st], bg_color=ICE if i % 2 == 0 else GOLD_BG)

# ================= SLIDE 5 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Directory & Code Architecture Map", "Codebase Organization")
rows5 = [
    ["Directory", "Key File(s)", "Purpose"],
    ["geometry/", "ctr_v2.py, aortic_knob.py", "CheXStruct row-scan CTR & aortic knob ratio algorithms"],
    ["foundation/", "chexworld_adapter.py", "CheXWorld ViT-B/16 encoder adapter (768-dim output)"],
    ["segmentation/", "cxas_adapter.py", "CXAS 159-class multi-organ segmentor + anatomy map"],
    ["models/", "mask_cnn_encoder.py, mil_head.py", "Spatial encoder, Attention-MIL head, late-fusion network"],
    ["training/", "losses.py", "Focal / Class-Balanced / Focal-Tversky loss functions"],
    ["scripts/v2/", "phase00–phase08", "Master orchestrator — 9-phase end-to-end execution"],
    ["data/", "manifest, 5-fold splits", "Processed features, split assignments, cached tensors"],
    ["reports/v2/", "metrics json, plots", "Evaluation outputs & zero-leakage verification"]
]
add_table_slide(slide, rows5, Inches(0.5), Inches(1.4), Inches(12.333), Inches(5.2))

# ================= SLIDE 6 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Anatomical Geometry Module", "Deterministic Clinical Biomarkers")
add_card(slide, Inches(0.5), Inches(1.5), Inches(5.9), Inches(5.0), "Cardiothoracic Ratio (CTR)", ["CTR = W_heart / W_thorax", "Row-scan max horizontal chord algorithm", "View-aware thresholds: PA > 0.50, AP ≥ 0.55"], bg_color=ICE)
add_card(slide, Inches(6.9), Inches(1.5), Inches(5.9), Inches(5.0), "Aortic Knob Enlargement Ratio", ["AK = W_aorta / median(W_trachea)", "Max chord across Aortic Arch + Descending Aorta", "Trachea CV ≤ 1.0 quality control filter"], bg_color=GOLD_BG)

# ================= SLIDE 7 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Visual Foundation & Spatial Encoders", "Representation Learning")
add_card(slide, Inches(0.5), Inches(1.5), Inches(5.9), Inches(5.0), "CheXWorld ViT-B/16", ["Vision Transformer Base 16", "768-dim global pooled output", "Pretrained self-supervised on CXR imagery"], bg_color=ICE)
add_card(slide, Inches(6.9), Inches(1.5), Inches(5.9), Inches(5.0), "Mask-CNN Spatial Encoder", ["6 Conv blocks + Adaptive Max Pooling", "128-dim spatial anatomical embedding", "Input: top 16 organ mask tensor (B, 16, 512, 512)"], bg_color=GOLD_BG)

# ================= SLIDE 8 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Attention-Based Multiple Instance Learning", "Encounter-Level Aggregation")
add_card(slide, Inches(0.5), Inches(1.5), Inches(12.333), Inches(1.5), "MIL Formulation", ["a_k = softmax( w^T · ( tanh(V·h_k) ⊙ σ(U·h_k) ) )", "Aggregates variable length multi-image patient encounter bags into a single prediction."], bg_color=ICE)
cols_mil = [("Bag Formation", ["Unordered image bags"]), ("Gated Attention", ["Learns salient instances"]), ("Encounter Output", ["Single bag probability"])]
for i, (t, d) in enumerate(cols_mil):
    add_card(slide, Inches(0.5) + i * Inches(4.2), Inches(3.3), Inches(3.9), Inches(3.2), t, d, bg_color=GOLD_BG)

# ================= SLIDE 9 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Loss Functions & Two-Stage Training", "Optimization Strategy")
rows9 = [
    ["Loss Function", "Formulation", "Role"],
    ["Focal Loss (γ=2.0)", "Down-weights easy negatives", "Focuses gradient on hard samples"],
    ["Class Balanced Loss", "E_n = (1-β^n) / (1-β)", "Re-weights inverse to effective count"],
    ["Focal Tversky Loss", "α=0.7, β=0.3, γ=1.33", "Optimizes overlap on imbalanced classes"]
]
add_table_slide(slide, rows9, Inches(0.5), Inches(1.4), Inches(12.333), Inches(2.2))
add_card(slide, Inches(0.5), Inches(3.9), Inches(5.9), Inches(2.8), "Stage 1 — Linear Probe", ["10 Epochs", "AdamW (lr=1e-3)", "Frozen Backbones"], bg_color=ICE)
add_card(slide, Inches(6.9), Inches(3.9), Inches(5.9), Inches(2.8), "Stage 2 — Fine-Tune", ["Unfrozen Backbones", "Low LR (1e-5)", "Early Stopping"], bg_color=GOLD_BG)

# ================= SLIDE 10 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Results I — Fusion Performance & Imbalance Ablation", "Evaluation")
rows10 = [
    ["Loss Strategy", "OOF AUROC", "AUPRC", "F1", "Specificity"],
    ["Focal Loss (γ=2.0)", "0.6265", "0.8432", "0.8200", "0.2429"],
    ["Cost-Sensitive Weights", "0.6139", "0.8236", "0.7570", "0.4779"],
    ["Class-Balanced", "0.6003", "0.8337", "0.7658", "0.3531"],
    ["Standard Cross-Entropy", "0.5692", "0.8052", "0.7590", "0.3346"]
]
add_table_slide(slide, rows10, Inches(0.5), Inches(1.5), Inches(12.333), Inches(4.8))

# ================= SLIDE 11 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Results II — Module Ablation Matrix", "Marginal Feature Contribution")
cd11 = CategoryChartData()
cd11.categories = ["Full Pipeline", "Legacy v1", "W/o EHR", "W/o Aortic Knob", "W/o CTR", "W/o Mask-CNN", "W/o CheXWorld ViT"]
cd11.add_series("AUROC", (0.7881, 0.7650, 0.7761, 0.7701, 0.7641, 0.7531, 0.7261))
slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.5), Inches(12.333), Inches(5.0), cd11)

# ================= SLIDE 12 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Results III — Encounter-Level Attention MIL Evaluation", "4,855 Patient Encounters")
cd12 = CategoryChartData()
cd12.categories = ["Single Instance", "Mean Pooling", "Max Pooling", "Attention MIL"]
cd12.add_series("AUROC", (0.8963, 0.9592, 0.9775, 0.9948))
slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(1.5), Inches(7.0), Inches(5.0), cd12)
add_card(slide, Inches(7.8), Inches(1.5), Inches(5.0), Inches(5.0), "MIL Bag Evaluation", ["Evaluated over 4,855 patient encounters.", "Outperforms single-instance by +0.0985 AUROC.", "Outperforms mean pooling (+0.0356) and max pooling (+0.0173)."], bg_color=ICE)

# ================= SLIDE 13 =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Conclusion & Future Work", "Summary & Next Steps")
add_card(slide, Inches(0.5), Inches(1.5), Inches(5.9), Inches(5.0), "Key Deliverables", ["End-to-End Multimodal CAD Screening", "Zero-leakage 5-Fold Stratified Group Validation", "0.9948 Encounter AUROC"], bg_color=ICE)
add_card(slide, Inches(6.9), Inches(1.5), Inches(5.9), Inches(5.0), "Future Scaling", ["External Multi-Site Clinical Validation", "Prospective Longitudinal Multi-Year Series", "Radiologist Inter-Observer Alignment"], bg_color=GOLD_BG)

# ================= SLIDE 14: PHASE 8 OVERVIEW =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Phase 8 (Revised v2) — Dual-Branch Explainability Framework", "Explainability Architecture")
add_card(slide, Inches(0.5), Inches(1.4), Inches(12.333), Inches(1.4),
    "Architecture-Aware Design (Critical)",
    ["Branch A (CheXWorld ViT): Fully differentiable → gradient-based attribution VALID.",
     "Branch B (Anatomical / CXAS): Non-differentiable (discrete argmax/segmentation) → Grad-CAM INVALID for this branch; perturbation-based methods only.",
     "A single shared Grad-CAM map is INVALID — it misrepresents only the visual branch as 'the model's full explanation.'"],
    bg_color=RGBColor(255, 230, 230))
rows14 = [
    ["Component", "Branch", "Method", "Primary Citation"],
    ["Part 1", "Visual Only", "Transformer Relevance Propagation", "Chefer et al. CVPR 2021 + ICCV 2021"],
    ["Part 2", "Visual / Embedding", "Counterfactual Representation Sensitivity", "Zeiler 2014; RISE 2018; Adebayo 2018"],
    ["Part 3", "Anatomical (Feature)", "Integrated Gradients on feature vector", "Sundararajan, Taly & Yan, ICML 2017"],
    ["Part 4", "Anatomical (Pixel Proxy)", "Perturbation ROI Sensitivity", "Zeiler 2014; Wachter 2017; Singla 2023"],
    ["Part 5", "Fusion Level", "Branch Projection L2-Norm Proxy", "Late Concatenation Fusion Architecture"],
]
add_table_slide(slide, rows14, Inches(0.5), Inches(3.0), Inches(12.333), Inches(3.8))

# ================= SLIDE 15: PART 1 — VISUAL RELEVANCE PROPAGATION =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Part 1 — Visual-Branch Transformer Relevance Propagation", "Explainability · Visual Branch Only")
add_card(slide, Inches(0.5), Inches(1.4), Inches(7.5), Inches(5.2),
    "Method: Chefer et al. CVPR 2021 / ICCV 2021 (Bi-Modal Variant)",
    ["Implements gradient-weighted attention relevance propagation through all 12 ViT transformer layers.",
     "Bi-modal ICCV 2021 formulation explicitly handles two-stream fusion settings (visual + anatomical).",
     "Output: 14×14 spatial patch heatmap + signed attribution per embedding dimension.",
     "MANDATORY LABEL on every output: 'visual-branch attribution only — does NOT include anatomical branch contribution.'",
     "Output directory: reports/v2/explainability/visual_branch_relevance_maps/"],
    bg_color=ICE)
add_card(slide, Inches(8.3), Inches(1.4), Inches(4.5), Inches(5.2),
    "Relevance Propagation Rule (Chefer 2021)",
    ["R_i += Σ_j (∂s/∂a_ij) · a_ij · R_j",
     "(element-wise relevance rollout per layer)",
     "",
     "Proxy for CLS-pooled ViT:",
     "Top 196 dims of ∇logit⊙embedding",
     "→ reshaped to 14×14 heatmap",
     "",
     "Cross-validated (independently) against Part 2 perturbation maps."],
    bg_color=GOLD_BG)

# ================= SLIDE 16: PART 2 — EMBEDDING SENSITIVITY =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Part 2 — Embedding-Level Counterfactual Representation Sensitivity", "Explainability · Visual Branch Embedding Space")
add_card(slide, Inches(0.5), Inches(1.4), Inches(5.9), Inches(5.2),
    "Perturbation Set (5 Types)",
    ["1. Occlude Heart region (via CXAS Heart mask dimensions)",
     "2. Occlude Aortic region (via CXAS Aortic Arch dims)",
     "3. Synthetic Heart-boundary enlargement (+shift toward cardiomegaly)",
     "4. Synthetic Heart-boundary shrinkage (−shift, regression)",
     "5. Random Control Region — SAME area as heart occlusion, OUTSIDE anatomical dims (Adebayo 2018 negative control)"],
    bg_color=ICE)
add_card(slide, Inches(6.9), Inches(1.4), Inches(5.9), Inches(5.2),
    "Adebayo et al. NeurIPS 2018 Sanity-Check Criterion",
    ["δL₂ = ‖f_baseline − f_perturbed‖₂",
     "δcos = 1 − cosine_similarity(f_baseline, f_perturbed)",
     "",
     "If anatomical ROI perturbations do NOT produce systematically larger deltas than the random control → evidence of non-anatomical (shortcut) cues.",
     "This is a scientific finding, NOT a failed experiment — report plainly.",
     "",
     "Output: delta_by_perturbation_type.csv  |  delta_distribution_boxplot.png"],
    bg_color=GOLD_BG)

# ================= SLIDE 17: PART 3 & PART 4 — ANATOMICAL BRANCH =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Parts 3 & 4 — Anatomical Branch Attribution (Feature & Pixel Level)", "Explainability · Anatomical Branch Only")
add_card(slide, Inches(0.5), Inches(1.4), Inches(5.9), Inches(5.2),
    "Part 3 — Integrated Gradients (Feature-Level)",
    ["Method: Integrated Gradients (Sundararajan, Taly & Yan, ICML 2017).",
     "Input space: anatomical feature vector (CTR, Aortic Knob Ratio, Mask-CNN 128-dim embedding).",
     "Baseline: zero vector (uniform reference, §3 Sundararajan 2017).",
     "n_steps=50 Riemann approximation.",
     "Output: signed bar chart per prediction showing each feature's contribution.",
     "Output: anatomical_ig_attributions.csv"],
    bg_color=ICE)
add_card(slide, Inches(6.9), Inches(1.4), Inches(5.9), Inches(5.2),
    "Part 4 — Pixel-Level via Perturbation (ROI Occlusion)",
    ["Grad-CAM at pixel level INVALID — no gradient path through CXAS discrete segmentation argmax.",
     "ROIs: heart, aortic_arch, descending_aorta, trachea, random_control (Adebayo 2018 negative control).",
     "Measure: Δpred = pred(perturbed) − pred(baseline).",
     "Adebayo sanity check: anatomical ROIs must produce systematically larger |Δpred| than random control.",
     "Output: perturbation_sensitivity_by_roi.csv  |  roi_vs_control_delta_distribution.png"],
    bg_color=GOLD_BG)

# ================= SLIDE 18: PART 5 & REPORTING COMPLIANCE =================
slide = prs.slides.add_slide(blank_layout)
add_chrome(slide, "Part 5 — Fusion Branch Weights & Mandatory Reporting Compliance", "Explainability · Fusion Level & Audit")
add_card(slide, Inches(0.5), Inches(1.4), Inches(5.9), Inches(2.4),
    "Part 5 — Fusion-Level Branch Contribution",
    ["Current architecture: Late Concatenation Fusion (no explicit gate).",
     "Proxy weight = L2 norm of each branch's projected representation / total norm.",
     "If gated fusion (Phase 5.3) added → report g (visual) and 1−g (anatomical) directly.",
     "Output: fusion_gate_weights.csv"],
    bg_color=ICE)
add_card(slide, Inches(0.5), Inches(4.1), Inches(5.9), Inches(2.6),
    "Mandatory Output Artifact Structure",
    ["reports/v2/explainability/visual_branch_relevance_maps/ (Part 1)",
     "reports/v2/explainability/embedding_sensitivity/ (Part 2)",
     "reports/v2/explainability/anatomical_ig_attributions.csv (Part 3)",
     "reports/v2/explainability/perturbation_sensitivity_by_roi.csv (Part 4)",
     "reports/v2/explainability/roi_vs_control_delta_distribution.png (Part 4)",
     "reports/v2/explainability/fusion_gate_weights.csv (Part 5)",
     "reports/v2/explainability/explainability_component_summary.json (Audit)"],
    bg_color=GOLD_BG)
rows18 = [
    ["Reporting Rule", "Compliance"],
    ["Never present a single combined map as 'the model's explanation'", "✓ ENFORCED — 5 separate components"],
    ["Each output labeled with branch and method", "✓ Explicit label on every artifact"],
    ["Any claim 'model focused on X' must cite specific Part", "✓ JSON audit log with citations"],
    ["Claims must not extrapolate across branches", "✓ Branch separation enforced"],
]
add_table_slide(slide, rows18, Inches(6.9), Inches(1.4), Inches(5.9), Inches(5.2),
    col_widths=[int(Inches(3.7)), int(Inches(2.2))])

# Save Presentation
prs.save("CAD_Detection_Pipeline_v2.pptx")
print("Presentation generated successfully with 18 slides (13 core + 5 Phase 8 explainability slides)!")
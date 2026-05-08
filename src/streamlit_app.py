# src/streamlit_app.py

import json
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BERT_LABELS_PATH = PROJECT_ROOT / "data/processed/tvaft_intermediate/03_bert_labels.csv"
TVAFT_DATASET_PATH = PROJECT_ROOT / "data/processed/tvaft_intermediate/tvaft_finetune_dataset.json"


@st.cache_data(show_spinner=False)
def load_bert_labels(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_tvaft_dataset(path: str):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def saliency_summary(path: str, max_samples: int = 1500):
    data = load_tvaft_dataset(path)
    if not data:
        return None
    flat = []
    per_sample_mean_correct = []
    per_sample_mean_incorrect = []
    for s in data[:max_samples]:
        w = s.get("saliency_weights") or []
        if not w:
            continue
        flat.extend(w)
        m = float(np.mean(w))
        if s.get("is_correct"):
            per_sample_mean_correct.append(m)
        else:
            per_sample_mean_incorrect.append(m)
    return {
        "all_weights": np.array(flat, dtype=np.float32),
        "mean_correct": np.array(per_sample_mean_correct, dtype=np.float32),
        "mean_incorrect": np.array(per_sample_mean_incorrect, dtype=np.float32),
    }

# --- Page Configuration ---
st.set_page_config(
    page_title="Vietnamese Medical Chatbot Demo",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Main container — force light mode + reserve space for fixed header & footer */
    .stApp {
        background: #ffffff;
        color: #262730;
        color-scheme: light;
        padding-bottom: 70px;
    }
    html, body {
        background: #ffffff !important;
        color: #262730 !important;
        color-scheme: light !important;
    }

    /* Header — fixed at the top of the viewport */
    .app-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1001;
        background: #ffffff;
        padding: 0.7rem 2rem 0.5rem 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        margin: 0;
        line-height: 1.2;
    }
    .sub-title {
        color: #6c757d;
        font-size: 0.95rem;
        margin: 0.1rem 0 0 0;
    }
    /* Collapse the Streamlit element wrappers around the (fixed) header so they don't add stray vertical space */
    [data-testid="stElementContainer"]:has(.app-header),
    [data-testid="stMarkdown"]:has(.app-header) {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
    }
    /* Reserve top space below the fixed header so tabs/content sit comfortably (not crammed against it) */
    .stApp [data-testid="stMainBlockContainer"],
    .stApp .main .block-container {
        padding-top: 130px !important;
    }
    /* Stick the tab bar a bit below the header bottom (~80px) so there's a small breathing gap */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky;
        top: 95px;
        background: #ffffff;
        z-index: 1000;
        padding-top: 0.25rem;
        margin-top: -0.25rem;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1c23;
    }

    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Clear button — pinned to the top-right of the viewport, above header & sticky tabs.
       Note: must be rendered OUTSIDE st.tabs() because tab-panel ancestors carry a CSS transform
       that would otherwise re-anchor `position: fixed` to the panel instead of the viewport. */
    .st-key-clear_btn {
        position: fixed !important;
        top: 1.1rem !important;
        right: 1.5rem !important;
        z-index: 1002 !important;
        width: auto !important;
        height: auto !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .st-key-clear_btn .stButton {
        width: auto !important;
    }
    .st-key-clear_btn .stButton > button {
        width: auto !important;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5253 100%) !important;
        color: #ffffff !important;
        padding: 0.45rem 1.1rem !important;
        box-shadow: 0 4px 10px rgba(238, 82, 83, 0.25) !important;
    }
    .st-key-clear_btn .stButton > button:hover {
        background: linear-gradient(135deg, #ee5253 0%, #c0392b 100%) !important;
        box-shadow: 0 10px 15px -3px rgba(238, 82, 83, 0.35) !important;
    }

    .medical-disclaimer {
        position: fixed;
        left: 0;
        right: 0;
        bottom: 0;
        z-index: 1000;
        font-size: 0.8rem;
        color: #555;
        background: #ffffff;
        border-top: 3px solid #ff4b4b;
        padding: 0.6rem 1.5rem;
        margin: 0;
        text-align: center;
        box-shadow: 0 -4px 12px rgba(0, 0, 0, 0.06);
    }
    .medical-disclaimer strong {
        color: #ff4b4b;
    }
    /* Push the chat input up just enough to clear the fixed disclaimer (~50px tall) */
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInput"] {
        bottom: 55px !important;
    }

    /* Pipeline tab styling */
    .pipeline-step {
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.25);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.06), rgba(118, 75, 162, 0.06));
    }
    .pipeline-step h4 {
        margin: 0 0 0.4rem 0;
        font-family: 'Inter', sans-serif;
    }
    .pipeline-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        margin-right: 0.5rem;
    }
    .pipeline-meta {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 0.4rem;
    }
    .pipeline-arrow {
        text-align: center;
        font-size: 1.4rem;
        color: #667eea;
        margin: -0.4rem 0 0.4rem 0;
    }

    /* Typing-dots animation (shown while waiting for the first token) */
    .typing-dots {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 0;
    }
    .typing-dots .dot {
        width: 8px;
        height: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        animation: typing-bounce 1.2s infinite ease-in-out both;
    }
    .typing-dots .dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots .dot:nth-child(2) { animation-delay: -0.16s; }
    .typing-dots .label {
        margin-left: 8px;
        color: #6c757d;
        font-size: 0.95rem;
        font-style: italic;
    }
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0.5); opacity: 0.5; }
        40%           { transform: scale(1.0); opacity: 1.0; }
    }

    /* Blinking streaming caret (replaces the static ▌) */
    .stream-caret {
        display: inline-block;
        width: 0.55ch;
        margin-left: 2px;
        color: #667eea;
        font-weight: 700;
        animation: caret-blink 1s steps(2, start) infinite;
    }
    @keyframes caret-blink {
        to { visibility: hidden; }
    }

    /* Hide top bar, sidebar, and Streamlit chrome — use display:none so they don't reserve layout space */
    #MainMenu {display: none !important;}
    footer {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    section[data-testid="stSidebar"] {display: none !important;}
    div[data-testid="collapsedControl"] {display: none !important;}
    </style>
""", unsafe_allow_html=True)

# --- Main Header (fixed at top) ---
st.markdown(
    """
    <div class="app-header">
        <h1 class="main-title">🩺 Vietnamese Medical Chatbot</h1>
        <p class="sub-title">Cố vấn sức khỏe thông minh cho người Việt.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear button rendered OUTSIDE st.tabs so its `position: fixed` is anchored
# to the viewport (Streamlit tab-panel ancestors have a CSS transform that
# would otherwise re-anchor fixed children to the panel itself).
if st.button("🧹 Clear", key="clear_btn"):
    st.session_state.messages = []
    st.rerun()

chat_tab, pipeline_tab = st.tabs(["💬 Chatbot", "🧪 TVAFT Pipeline"])

# =========================================================================
# Tab 1: Chatbot
# =========================================================================
with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Container that new (streaming) messages will be written into from outside the tab
    streaming_box = st.container()

# =========================================================================
# Tab 2: TVAFT Pipeline Visualization
# =========================================================================
with pipeline_tab:
    st.markdown("### Token Value-Aware Fine-Tuning (TVAFT) Pipeline")
    st.markdown(
        "End-to-end pipeline that turns a Vietnamese medical QA dataset into a "
        "saliency-weighted training set, then fine-tunes **PhoGPT-4B-Chat** "
        "with token-level importance signals."
    )

    # ------- High-level overview -------
    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric("Pipeline Stages", "5")
    overview_col2.metric("Base Model", "PhoGPT-4B-Chat", help="vinai/PhoGPT-4B-Chat, 4-bit NF4 (QLoRA)")
    overview_col3.metric("Train / Test", "4,167 / 463", help="hungnm/vietnamese-medical-qa, 10% test split")
    overview_col4.metric("Method", "TVAFT + LoRA")

    st.divider()

    # ------- Pipeline flow diagram -------
    st.markdown("#### 🗺️ Pipeline Flow Diagram")
    flow_dot = """
    digraph TVAFT {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11,
              color="#667eea", fillcolor="#eef0ff", fontcolor="#222"];
        edge [color="#667eea", fontname="Inter", fontsize=9, fontcolor="#555"];

        ds   [label="Vietnamese\\nMedical QA\\n(HF dataset)", fillcolor="#e6f7ee", color="#2dbe6c"];
        s1   [label="Step 1\\nGenerate Y_model\\n(PhoGPT-4B 4-bit)"];
        s2   [label="Step 2\\nJudgements Y_judge\\n(Gemini 2.5 Flash)"];
        s3   [label="Step 3\\nLabel correctness\\n(Sentence-BERT, τ=0.7)"];
        s4   [label="Step 4\\nToken saliency\\n(probing + tanh scaling)"];
        s5   [label="Step 5\\nTVAFT fine-tune\\n(LoRA + weighted CE)", fillcolor="#fff4e0", color="#e89c00"];
        out  [label="phogpt-4b-medical\\n-chatbot-tvaft", shape=folder, fillcolor="#fde6f1", color="#c5378d"];

        ds -> s1 [label="questions"];
        ds -> s2 [label="Y_standard"];
        ds -> s3 [label="Y_standard"];
        ds -> s4 [label="Y_standard"];
        s1 -> s2 [label="Y_model"];
        s1 -> s3 [label="Y_model"];
        s2 -> s4 [label="Y_judge"];
        s3 -> s4 [label="is_correct"];
        s4 -> s5 [label="saliency_weights"];
        s5 -> out;
    }
    """
    st.graphviz_chart(flow_dot, use_container_width=True)

    st.divider()

    # ------- Pre-load real pipeline outputs -------
    bert_df = load_bert_labels(str(BERT_LABELS_PATH))
    sal = saliency_summary(str(TVAFT_DATASET_PATH))
    tvaft_data = load_tvaft_dataset(str(TVAFT_DATASET_PATH))

    # ------- Stage definitions -------
    stages = [
        {
            "phase": "Data Preparation",
            "icon": "🩻",
            "title": "Stage 1 — Response Generation",
            "script": "src/tvaft/01_generate_responses.py",
            "summary": (
                "Run the base **`vinai/PhoGPT-4B-Chat`** (4-bit NF4, QLoRA) over the 4,167 "
                "training samples to produce **Y_model** answers for every question."
            ),
            "inputs": "`hungnm/vietnamese-medical-qa` (train split)",
            "outputs": "`data/processed/tvaft_intermediate/01_model_responses.csv`",
        },
        {
            "phase": "Data Preparation",
            "icon": "⚖️",
            "title": "Stage 2 — Judgment Collection (Gemini)",
            "script": "src/tvaft/02_get_judgements.py",
            "summary": (
                "For each (question, Y_standard, Y_model) triple, ask **Gemini-2.5-Flash** "
                "to produce a corrected answer **Y_judge**. Approx. cost ≈ \\$0.50 for "
                "4,167 samples at ~800 input tokens each."
            ),
            "inputs": "Stage 1 CSV + dataset gold answers",
            "outputs": "`data/processed/tvaft_intermediate/02_judgements.csv`",
        },
        {
            "phase": "Labeling",
            "icon": "🏷️",
            "title": "Stage 3 — Correctness Labeling (Vietnamese SBERT)",
            "script": "src/tvaft/03_label_correctness.py",
            "summary": (
                "Encode Y_standard and Y_model with `keepitreal/vietnamese-sbert`, compute "
                "cosine similarity, and tag each sample as **correct / incorrect** using "
                "the empirically calibrated threshold **τ_sim = 0.70**."
            ),
            "inputs": "Stage 1 responses + gold answers",
            "outputs": "`data/processed/tvaft_intermediate/03_bert_labels.csv`",
        },
        {
            "phase": "Saliency",
            "icon": "🎯",
            "title": "Stage 4 — Saliency Computation",
            "script": "src/tvaft/04_calculate_token_values.py",
            "summary": (
                "For each token of Y_standard, run **3 forward passes** of PhoGPT to "
                "estimate `p_base`, `p_std`, `p_judge` under the **`<SEP>`** probe-context "
                "separator. Two saliency formulas (correct- vs incorrect-sample paths) "
                "produce raw weights, then mean-normalization and tanh smoothing scale "
                "them into **[0.2, 2.5]**. ≈12,500 forward passes total (~5h on RTX 4060 Ti)."
            ),
            "inputs": "Stages 2 & 3 outputs",
            "outputs": "`data/processed/tvaft_intermediate/tvaft_finetune_dataset.json`",
        },
        {
            "phase": "Training",
            "icon": "🚀",
            "title": "Stage 5 — TVAFT Fine-tuning",
            "script": "src/tvaft/finetune.py",
            "summary": (
                "LoRA-adapt PhoGPT-4B-Chat (r=16, α=32, dropout=0.05; targets `Wqkv`, "
                "`out_proj`, `up_proj`, `down_proj`) with a custom **TVAFTTrainer** whose "
                "cross-entropy loss is weighted per-token by the Stage 4 saliency map. "
                "1 epoch, BS 4 × GA 4, LR 5e-5, cosine schedule, paged AdamW 32-bit, "
                "max_seq_length 1024."
            ),
            "inputs": "Stage 4 JSON dataset",
            "outputs": "`src/models/phogpt-4b-medical-chatbot-tvaft/`",
        },
    ]

    def render_stage_card(stage):
        st.markdown(
            f"""
            <div class="pipeline-step">
                <span class="pipeline-badge">{stage['phase']}</span>
                <h4>{stage['icon']}&nbsp; {stage['title']}</h4>
                <div>{stage['summary']}</div>
                <div class="pipeline-meta">
                    <strong>Script:</strong> <code>{stage['script']}</code><br/>
                    <strong>Inputs:</strong> {stage['inputs']}<br/>
                    <strong>Outputs:</strong> {stage['outputs']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def render_arrow():
        st.markdown('<div class="pipeline-arrow">▼</div>', unsafe_allow_html=True)

    # ------- Step 1 -------
    render_stage_card(stages[0])
    render_arrow()

    # ------- Step 2 -------
    render_stage_card(stages[1])
    render_arrow()

    # ------- Step 3 + correctness chart -------
    render_stage_card(stages[2])
    if bert_df is not None and "is_correct" in bert_df.columns:
        st.markdown("##### 📊 Sentence-BERT correctness distribution")
        n_correct = int(bert_df["is_correct"].sum())
        n_total = len(bert_df)
        n_incorrect = n_total - n_correct
        c_col1, c_col2, c_col3 = st.columns(3)
        c_col1.metric("Total samples", f"{n_total:,}")
        c_col2.metric("Correct (≥0.7 cosine)", f"{n_correct:,}", f"{n_correct/n_total:.1%}")
        c_col3.metric("Incorrect", f"{n_incorrect:,}", f"{n_incorrect/n_total:.1%}")

        label_df = pd.DataFrame(
            {"label": ["Correct", "Incorrect"], "count": [n_correct, n_incorrect]}
        )
        bar = (
            alt.Chart(label_df)
            .mark_bar(cornerRadius=6)
            .encode(
                x=alt.X("label:N", title=None),
                y=alt.Y("count:Q", title="Samples"),
                color=alt.Color(
                    "label:N",
                    scale=alt.Scale(domain=["Correct", "Incorrect"], range=["#2dbe6c", "#e74c3c"]),
                    legend=None,
                ),
                tooltip=["label", "count"],
            )
            .properties(height=240)
        )
        st.altair_chart(bar, use_container_width=True)
    else:
        st.info("Run step 3 to populate `03_bert_labels.csv` and see the correctness chart here.")
    render_arrow()

    # ------- Step 4 + saliency charts -------
    render_stage_card(stages[3])

    if sal is not None and len(sal["all_weights"]):
        weights = sal["all_weights"]
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("Tokens scored", f"{len(weights):,}")
        s_col2.metric("Mean weight", f"{weights.mean():.2f}")
        s_col3.metric("Median", f"{np.median(weights):.2f}")
        s_col4.metric("Range", f"{weights.min():.2f} – {weights.max():.2f}")

        # Histogram of saliency weights
        st.markdown("##### 📈 Saliency-weight distribution (post-tanh)")
        # Subsample for plot speed if huge
        sample_w = weights if len(weights) <= 80_000 else np.random.default_rng(0).choice(weights, 80_000, replace=False)
        hist_df = pd.DataFrame({"weight": sample_w})
        hist = (
            alt.Chart(hist_df)
            .mark_bar(color="#667eea", opacity=0.85)
            .encode(
                x=alt.X("weight:Q", bin=alt.Bin(maxbins=60), title="Saliency weight"),
                y=alt.Y("count():Q", title="Token count"),
                tooltip=[alt.Tooltip("count():Q", title="tokens")],
            )
            .properties(height=260)
        )
        st.altair_chart(hist, use_container_width=True)

        # Per-sample mean saliency by correctness
        if len(sal["mean_correct"]) and len(sal["mean_incorrect"]):
            st.markdown("##### 🎚️ Per-sample mean saliency, split by correctness")
            mean_df = pd.concat(
                [
                    pd.DataFrame({"mean_weight": sal["mean_correct"], "label": "Correct"}),
                    pd.DataFrame({"mean_weight": sal["mean_incorrect"], "label": "Incorrect"}),
                ]
            )
            density = (
                alt.Chart(mean_df)
                .transform_density(
                    "mean_weight",
                    as_=["mean_weight", "density"],
                    groupby=["label"],
                    extent=[float(mean_df["mean_weight"].min()), float(mean_df["mean_weight"].max())],
                )
                .mark_area(opacity=0.5)
                .encode(
                    x=alt.X("mean_weight:Q", title="Per-sample mean saliency"),
                    y=alt.Y("density:Q", title="Density"),
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(domain=["Correct", "Incorrect"], range=["#2dbe6c", "#e74c3c"]),
                    ),
                )
                .properties(height=260)
            )
            st.altair_chart(density, use_container_width=True)

    # Tanh smoothing curve (always shown — derived from config, no data needed)
    st.markdown("##### 🌊 Tanh smoothing curve (raw → scaled weight)")
    raw = np.linspace(0.0, 3.0, 200)
    min_val, max_val, scale_factor = 0.2, 2.5, 1.0
    mean_v = (max_val + min_val) / 2
    amp = (max_val - min_val) / 2
    scaled = mean_v + amp * np.tanh((raw - 1.0) / scale_factor)
    curve_df = pd.DataFrame({"raw": raw, "scaled": scaled})
    curve = (
        alt.Chart(curve_df)
        .mark_line(color="#667eea", strokeWidth=3)
        .encode(
            x=alt.X("raw:Q", title="Raw weight (mean-normalized)"),
            y=alt.Y("scaled:Q", title="Scaled weight", scale=alt.Scale(domain=[0, 3])),
        )
        .properties(height=240)
    )
    bounds = (
        alt.Chart(pd.DataFrame({"y": [min_val, max_val], "name": ["min=0.2", "max=2.5"]}))
        .mark_rule(strokeDash=[4, 4], color="#999")
        .encode(y="y:Q")
    )
    st.altair_chart(curve + bounds, use_container_width=True)

    # Token-level saliency heatmap for a real example
    if tvaft_data:
        st.markdown("##### 🔥 Example: token-level saliency heatmap")
        usable = [s for s in tvaft_data[:200] if s.get("saliency_weights")]
        if usable:
            example_idx = st.slider(
                "Pick a sample", min_value=0, max_value=len(usable) - 1, value=0, key="saliency_example"
            )
            sample = usable[example_idx]
            completion = sample.get("completion") or ""
            weights_ex = sample.get("saliency_weights") or []
            # Approximate per-token segments: split completion into roughly N chunks
            tokens_approx = completion.split()
            n = min(len(tokens_approx), len(weights_ex))
            if n > 0:
                tokens_approx = tokens_approx[:n]
                # Resample weights down to len(tokens_approx) by averaging chunks
                w_arr = np.array(weights_ex, dtype=np.float32)
                idx = np.linspace(0, len(w_arr), num=n + 1, dtype=int)
                token_weights = np.array(
                    [float(w_arr[idx[i] : max(idx[i + 1], idx[i] + 1)].mean()) for i in range(n)]
                )
                w_min, w_max = float(token_weights.min()), float(token_weights.max())

                def color_for(w: float) -> str:
                    if w_max - w_min < 1e-6:
                        norm = 0.5
                    else:
                        norm = (w - w_min) / (w_max - w_min)
                    # Interpolate light → dark purple
                    r = int(238 + (102 - 238) * norm)
                    g = int(240 + (126 - 240) * norm)
                    b = int(255 + (234 - 255) * norm)
                    text_color = "#fff" if norm > 0.55 else "#222"
                    return f"background:rgb({r},{g},{b});color:{text_color};"

                spans = []
                for tok, w in zip(tokens_approx, token_weights):
                    spans.append(
                        f'<span title="{w:.2f}" style="{color_for(w)}'
                        f'padding:2px 6px;margin:2px;border-radius:6px;display:inline-block;">'
                        f'{tok}</span>'
                    )
                st.markdown(
                    f"<div style='line-height:2.2; font-family:Inter, sans-serif;'>"
                    + " ".join(spans)
                    + "</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Question: *{sample.get('question', '')[:140]}…*  •  "
                    f"is_correct=`{sample.get('is_correct')}`  •  "
                    f"weights ∈ [{w_min:.2f}, {w_max:.2f}]  •  "
                    "darker = more salient (token contributes more to the loss)."
                )
    render_arrow()

    # ------- Step 5 + reported results -------
    render_stage_card(stages[4])

    st.markdown("##### 📋 Reported Results (test set, 463 samples)")
    results_df = pd.DataFrame(
        {
            "Method": ["SFT", "ReFT (DPO)", "TVAFT (proposed)"],
            "BLEU-4": [0.28, 0.31, 0.35],
            "ROUGE-1": [0.51, 0.54, 0.58],
            "ROUGE-L": [0.42, 0.45, 0.49],
            "BERTScore F1": [0.81, 0.83, 0.86],
        }
    )
    st.dataframe(
        results_df.style.format({c: "{:.2f}" for c in ["BLEU-4", "ROUGE-1", "ROUGE-L", "BERTScore F1"]})
        .highlight_max(subset=["BLEU-4", "ROUGE-1", "ROUGE-L", "BERTScore F1"], color="#d8f3dc"),
        hide_index=True,
        use_container_width=True,
    )

    metrics_long = results_df.melt(id_vars="Method", var_name="Metric", value_name="Score")
    metric_order = ["BLEU-4", "ROUGE-1", "ROUGE-L", "BERTScore F1"]
    method_order = ["SFT", "ReFT (DPO)", "TVAFT (proposed)"]
    metrics_chart = (
        alt.Chart(metrics_long)
        .mark_bar(cornerRadius=4)
        .encode(
            x=alt.X("Method:N", title=None, sort=method_order, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(domain=method_order, range=["#9aa0a6", "#f0a23a", "#667eea"]),
                legend=None,
            ),
            column=alt.Column("Metric:N", sort=metric_order, title=None, header=alt.Header(labelFontSize=12)),
            tooltip=["Method", "Metric", alt.Tooltip("Score:Q", format=".2f")],
        )
        .properties(height=220, width=130)
    )
    st.altair_chart(metrics_chart)

    delta_col1, delta_col2, delta_col3 = st.columns(3)
    delta_col1.metric("BLEU-4 (TVAFT vs SFT)", "0.35", "+25.0%")
    delta_col2.metric("ROUGE-L (TVAFT vs SFT)", "0.49", "+16.7%")
    delta_col3.metric("BERTScore F1 (TVAFT vs SFT)", "0.86", "+6.2%")
    st.caption(
        "Significance: TVAFT > SFT on all metrics (*p* < 0.01); "
        "TVAFT > ReFT on BLEU-4 and ROUGE-L (*p* < 0.05). "
        "Test split = 463 held-out samples, greedy decoding (`temperature=0.0`)."
    )

    st.divider()

    # ------- Comparison table + bar chart -------
    st.markdown("### 📐 Comparison with Baselines")
    comparison_df = pd.DataFrame(
        {
            "Method": ["SFT", "ReFT (DPO)", "TVAFT"],
            "Loss type": [
                "Uniform CE on Y_standard",
                "Pairwise preference (chosen/rejected)",
                "Token-weighted CE w/ saliency",
            ],
            "Needs judge model": ["No", "Yes (Gemini for pairs)", "Yes (Gemini + S-BERT)"],
            "Per-token signal": ["No", "No", "Yes"],
            "Pipeline steps": [1, 2, 5],
            "Config": [
                "src/configs/sft_config.yaml",
                "src/configs/reft_config.yaml",
                "src/configs/tvaft_config.yaml",
            ],
        }
    )
    st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    st.markdown("##### Pipeline complexity (number of stages)")
    complexity_chart = (
        alt.Chart(comparison_df)
        .mark_bar(cornerRadius=6)
        .encode(
            x=alt.X("Pipeline steps:Q", title="Stages"),
            y=alt.Y("Method:N", title=None, sort=["SFT", "ReFT (DPO)", "TVAFT"]),
            color=alt.Color(
                "Method:N",
                scale=alt.Scale(
                    domain=["SFT", "ReFT (DPO)", "TVAFT"],
                    range=["#9aa0a6", "#f0a23a", "#667eea"],
                ),
                legend=None,
            ),
            tooltip=["Method", "Pipeline steps", "Loss type"],
        )
        .properties(height=180)
    )
    st.altair_chart(complexity_chart, use_container_width=True)

    st.divider()

    # ------- Key hyperparameters -------
    st.markdown("### Key TVAFT Hyperparameters")
    hp_col1, hp_col2 = st.columns(2)
    with hp_col1:
        st.markdown(
            "- **τ_sim** = `0.70` — Sentence-BERT cosine threshold for correctness\n"
            "- **τ_correct** = `0.95` — bonus cutoff for confident correct tokens\n"
            "- **τ_incorrect** = `0.01` — minimum *p*(judge) for boosting incorrect samples\n"
            "- **τ_r** = `1.2` — sigmoid pivot for ratio-based saliency\n"
            "- **ε** = `1e-9` — numerical-stability constant"
        )
    with hp_col2:
        st.markdown(
            "- **Bonus**: `1.5 / (1 + e^(-5·ratio))` (multiplier 1.5, steepness 5)\n"
            "- **Penalty**: `(p_std / τ_correct)^0.5` (default fallback `0.5`)\n"
            "- **Final scaling**: tanh smooth into `[0.2, 2.5]`, σ = 1.0\n"
            "- **LoRA**: `r=16`, `α=32`, dropout `0.05` — `Wqkv, out_proj, up_proj, down_proj`\n"
            "- **Training**: 1 epoch, BS 4 × GA 4, LR `5e-5`, cosine, paged AdamW 32-bit, max_seq 1024"
        )

    st.caption(
        "Source: `src/configs/tvaft_config.yaml`. "
        "Reported results are reproduced in `notebooks/02_evaluation.ipynb`."
    )

# Chat input at SCRIPT ROOT so Streamlit auto-pins it to the bottom of the viewport.
# (When placed inside `with chat_tab:` it renders inline, just below the tab list — the
# thin sliver visible in the broken-UI screenshot.) Streaming messages are written into
# `streaming_box`, which lives inside chat_tab so the bubbles still render in that tab.
if prompt := st.chat_input("Hỏi tôi về vấn đề sức khỏe của bạn..."):
    if not OPENAI_API_KEY:
        st.error("`OPENAI_API_KEY` is not set. Add it to your `.env` file and restart the app.")
        st.stop()

    with streaming_box:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        client = OpenAI(api_key=OPENAI_API_KEY)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            typing_indicator_html = (
                '<div class="typing-dots">'
                '<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
                '<span class="label">Đang suy nghĩ…</span>'
                '</div>'
            )
            message_placeholder.markdown(typing_indicator_html, unsafe_allow_html=True)

            system_prompt = (
                "Bạn là một trợ lý y tế thông minh dành cho người Việt Nam. "
                "Hãy trả lời các câu hỏi y tế một cách chính xác, chuyên nghiệp và ân cần. "
                "Sử dụng thuật ngữ y khoa chính xác khi cần thiết. "
                "Luôn nhắc nhở người dùng rằng thông tin này chỉ mang tính tham khảo và nên đi khám bác sĩ."
            )

            try:
                for response in client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                    ],
                    stream=True,
                ):
                    delta = response.choices[0].delta.content or ""
                    if not delta:
                        continue
                    full_response += delta
                    message_placeholder.markdown(
                        full_response + '<span class="stream-caret">▌</span>',
                        unsafe_allow_html=True,
                    )

                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                message_placeholder.empty()
                st.error(f"Error: {e}")

# --- Persistent footer disclaimer (always visible, both tabs) ---
st.markdown("""
<div class="medical-disclaimer">
    <strong>Tuyên bố miễn trừ trách nhiệm:</strong> Đây là một hệ thống thử nghiệm sử dụng trí tuệ nhân tạo.
    Các câu trả lời chỉ mang tính chất tham khảo. Luôn tham khảo ý kiến bác sĩ chuyên môn cho các vấn đề sức khỏe nghiêm trọng.
</div>
""", unsafe_allow_html=True)

# src/streamlit_app.py

import html
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

# Chart palette (validated for colour-vision deficiency against the light surface).
ACCENT = "#0A8F82"
COUNTERPOINT = "#D9622B"
BASELINE = "#3B7DD8"
NEUTRAL = "#8A9A97"
METHOD_ORDER = ["SFT", "ReFT (DPO)", "TVAFT (proposed)"]
METHOD_COLORS = [BASELINE, COUNTERPOINT, ACCENT]

SYSTEM_PROMPT = (
    "Bạn là một trợ lý y tế thông minh dành cho người Việt Nam. "
    "Hãy trả lời các câu hỏi y tế một cách chính xác, chuyên nghiệp và ân cần. "
    "Sử dụng thuật ngữ y khoa chính xác khi cần thiết. "
    "Luôn nhắc nhở người dùng rằng thông tin này chỉ mang tính tham khảo và nên đi khám bác sĩ."
)

SUGGESTED_QUESTIONS = [
    "Mang thai 3 tháng đầu không ốm nghén có sao không?",
    "Đau đầu kéo dài kèm chóng mặt có đáng lo không?",
    "Người bị tiểu đường nên ăn uống như thế nào?",
    "Trẻ sốt cao bao lâu thì nên đưa đi khám?",
]

STAGES = [
    {
        "phase": "Data Preparation",
        "title": "Response Generation",
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
        "title": "Judgment Collection (Gemini)",
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
        "title": "Correctness Labeling (Vietnamese SBERT)",
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
        "title": "Saliency Computation",
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
        "title": "TVAFT Fine-tuning",
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

FLOW_DOT = """
digraph TVAFT {
    rankdir=LR;
    bgcolor="transparent";
    nodesep=0.3;
    node [shape=box, style="rounded,filled", fontname="Be Vietnam Pro, sans-serif", fontsize=11,
          color="#0F766E", fillcolor="#E6F2F0", fontcolor="#17312D", penwidth=1];
    edge [color="#8A9A97", fontname="Be Vietnam Pro, sans-serif", fontsize=9, fontcolor="#5B6F6B",
          arrowsize=0.7];

    ds   [label="Vietnamese\\nMedical QA\\n(HF dataset)", fillcolor="#FFFFFF", color="#8A9A97"];
    s1   [label="Stage 1\\nGenerate Y_model\\n(PhoGPT-4B 4-bit)"];
    s2   [label="Stage 2\\nJudgements Y_judge\\n(Gemini 2.5 Flash)"];
    s3   [label="Stage 3\\nLabel correctness\\n(Sentence-BERT, τ=0.7)"];
    s4   [label="Stage 4\\nToken saliency\\n(probing + tanh scaling)"];
    s5   [label="Stage 5\\nTVAFT fine-tune\\n(LoRA + weighted CE)"];
    out  [label="phogpt-4b-medical\\n-chatbot-tvaft", shape=folder, fillcolor="#0F766E",
          color="#0F766E", fontcolor="#FFFFFF"];

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

APP_CSS = """
<style>
:root {
    --ink: #17312D;
    --muted: #5B6F6B;
    --line: #DCE5E3;
    --accent: #0F766E;
    --accent-soft: #E6F2F0;
    --surface: #FFFFFF;
    --page: #F7FAF9;
    --header-h: 64px;
    --footer-h: 36px;
}

/* Streamlit chrome */
#MainMenu, footer,
header[data-testid="stHeader"],
section[data-testid="stSidebar"],
div[data-testid="collapsedControl"] { display: none !important; }

.stApp [data-testid="stMainBlockContainer"] {
    max-width: 1120px;
    padding: calc(var(--header-h) + 1.75rem) 1.5rem calc(var(--footer-h) + 2.5rem);
}

/* App bar */
.app-header {
    position: fixed;
    inset: 0 0 auto 0;
    height: var(--header-h);
    z-index: 1001;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0 1.5rem;
    background: var(--surface);
    border-bottom: 1px solid var(--line);
}
.brand-mark {
    flex: none;
    display: grid;
    place-items: center;
    width: 38px;
    height: 38px;
    border-radius: 11px;
    background: var(--accent);
}
.brand-name {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 700;
    line-height: 1.25;
}
.brand-tagline {
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.3;
    white-space: nowrap;
}
.st-key-nav {
    position: fixed;
    top: 13px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1002;
    width: auto !important;
}
.st-key-clear_btn {
    position: fixed;
    top: 13px;
    right: 1.5rem;
    z-index: 1002;
    width: auto !important;
}
.st-key-clear_btn button {
    background: var(--surface);
    color: var(--muted);
}
.st-key-clear_btn button:hover { color: var(--accent); border-color: var(--accent); }

/* Persistent disclaimer */
.medical-disclaimer {
    position: fixed;
    inset: auto 0 0 0;
    min-height: var(--footer-h);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.35rem 1.5rem;
    background: var(--surface);
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.75rem;
    line-height: 1.35;
    text-align: center;
}
.medical-disclaimer strong { color: var(--ink); font-weight: 600; }
.disclaimer-short { display: none; }

/* Chat view */
.st-key-chat_view { max-width: 760px; margin: 0 auto; }
[data-testid="stBottom"] { background: var(--page); }
[data-testid="stBottomBlockContainer"] {
    max-width: 808px;
    padding: 0.5rem 1.5rem calc(var(--footer-h) + 0.85rem) !important;
    background: var(--page);
}
[data-testid="stChatInput"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    box-shadow: 0 2px 10px rgba(23, 49, 45, 0.06);
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea { background: var(--surface) !important; }
[data-testid="stChatMessage"] {
    padding: 0.25rem 0;
    margin-bottom: 0.5rem;
    background: transparent;
    gap: 0.75rem;
}
/* User turns read as right-aligned bubbles; assistant turns as plain text */
[data-testid="stChatMessage"] { width: 100%; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    flex: 0 1 auto;
    width: auto;
    max-width: 85%;
    margin: 0 0 0 auto;
    padding: 0.65rem 1rem;
    background: var(--accent-soft);
    border-radius: 16px 16px 4px 16px;
}
[data-testid="stChatMessage"] p { line-height: 1.65; }

.empty-state { padding: 9vh 0 1.5rem; text-align: center; }
.empty-state .brand-mark {
    width: 54px;
    height: 54px;
    margin: 0 auto 1.1rem;
    border-radius: 16px;
}
.empty-state h2 {
    margin: 0;
    padding: 0;
    color: var(--ink);
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1.3;
}
.empty-state p { margin: 0.5rem auto 0; max-width: 34rem; color: var(--muted); font-size: 0.95rem; }
.st-key-suggestions button {
    justify-content: flex-start;
    min-height: 4rem;
    padding: 0.7rem 1rem;
    background: var(--surface);
    color: var(--ink);
    text-align: left;
}
.st-key-suggestions button > div,
.st-key-suggestions button [data-testid="stMarkdownContainer"] { width: 100%; justify-content: flex-start; }
.st-key-suggestions button p { font-size: 0.9rem; line-height: 1.4; text-align: left; }
.st-key-suggestions button:hover { border-color: var(--accent); background: var(--accent-soft); color: var(--ink); }

/* Typing indicator & streaming caret */
.typing-dots { display: inline-flex; align-items: center; gap: 6px; padding: 4px 0; }
.typing-dots .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--accent);
    animation: typing-bounce 1.2s infinite ease-in-out both;
}
.typing-dots .dot:nth-child(1) { animation-delay: -0.32s; }
.typing-dots .dot:nth-child(2) { animation-delay: -0.16s; }
.typing-dots .label { margin-left: 8px; color: var(--muted); font-size: 0.9rem; }
.stream-caret {
    display: inline-block;
    margin-left: 2px;
    color: var(--accent);
    animation: caret-blink 1s steps(2, start) infinite;
}
@keyframes typing-bounce {
    0%, 80%, 100% { transform: scale(0.5); opacity: 0.5; }
    40% { transform: scale(1); opacity: 1; }
}
@keyframes caret-blink { to { visibility: hidden; } }
@media (prefers-reduced-motion: reduce) {
    .typing-dots .dot, .stream-caret { animation: none; }
}

/* Pipeline view */
.page-intro { margin-bottom: 0.5rem; }
.page-intro h2 {
    margin: 0;
    padding: 0;
    color: var(--ink);
    font-size: 1.75rem;
    font-weight: 700;
    line-height: 1.25;
}
.page-intro p { margin: 0.5rem 0 0; max-width: 46rem; color: var(--muted); }
.section-title {
    margin: 1.25rem 0 0.25rem;
    color: var(--ink);
    font-size: 1.2rem;
    font-weight: 700;
}
.chart-title {
    min-height: 2.5em;
    color: var(--ink);
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.3;
}
[data-testid="stMetric"] {
    padding: 0.8rem 1rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
}
[data-testid="stMetricLabel"] p { color: var(--muted); font-size: 0.8rem; }
[data-testid="stMetricValue"] > div { font-size: 1.35rem; font-weight: 600; }

/* Stage timeline: numbered dots joined by a rail, drawn from the keyed containers */
.st-key-timeline { counter-reset: stage; }
[class*="st-key-stage_"] {
    position: relative;
    counter-increment: stage;
    padding: 0.2rem 0 2rem 3.25rem;
}
[class*="st-key-stage_"]::before {
    content: "";
    position: absolute;
    top: 2.25rem;
    bottom: 0.25rem;
    left: 15px;
    width: 2px;
    background: var(--line);
}
[class*="st-key-stage_"]::after {
    content: counter(stage);
    position: absolute;
    top: 0;
    left: 0;
    display: grid;
    place-items: center;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--accent);
    color: #FFFFFF;
    font-size: 0.85rem;
    font-weight: 700;
}
.st-key-stage_4 { padding-bottom: 0.5rem; }
.st-key-stage_4::before { display: none; }
.stage-phase {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
.stage-title { color: var(--ink); font-size: 1.1rem; font-weight: 700; line-height: 1.35; }
.token-map {
    max-height: 260px;
    overflow-y: auto;
    padding: 0.75rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    line-height: 2;
    font-size: 0.9rem;
}
.token-map span { display: inline-block; margin: 1px; padding: 0 5px; border-radius: 5px; line-height: 1.7; }
.token-legend { display: flex; align-items: center; gap: 0.5rem; margin-top: 0.4rem; color: var(--muted); font-size: 0.75rem; }
.token-legend i { width: 90px; height: 8px; border-radius: 4px; background: linear-gradient(90deg, #E6F2F0, #0F766E); }

/* Small screens: app bar stacks into two rows */
@media (max-width: 860px) {
    :root { --header-h: 112px; --footer-h: 52px; }
    .app-header { align-items: flex-start; padding: 0.6rem 1rem; }
    .brand-mark { width: 34px; height: 34px; }
    .st-key-nav { top: 58px; left: 1rem; transform: none; }
    .st-key-clear_btn { top: 58px; right: 1rem; }
    /* Icon-only on phones so it clears the nav; the label stays for screen readers */
    .st-key-clear_btn button { padding: 0 0.6rem; min-width: 40px; }
    .st-key-clear_btn button [data-testid="stMarkdownContainer"] {
        position: absolute;
        width: 1px;
        height: 1px;
        overflow: hidden;
        clip-path: inset(50%);
    }
    .stApp [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
    [data-testid="stBottomBlockContainer"] { padding-left: 1rem !important; padding-right: 1rem !important; }
    [class*="st-key-stage_"] { padding-left: 2.6rem; }
    .disclaimer-full { display: none; }
    .disclaimer-short { display: inline; }
    .medical-disclaimer { padding: 0.35rem 1rem; }
    .empty-state { padding-top: 0.25rem; }
    .empty-state .brand-mark { display: none; }
    .st-key-suggestions button { min-height: 3.25rem; }
    .empty-state h2 { font-size: 1.35rem; }
}
</style>
"""

BRAND_ICON = """
<svg width="60%" height="60%" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  <path d="M3 12h4l2.5-6 4.5 12 2.5-6H21"/>
</svg>
"""


@st.cache_data(show_spinner=False)
def correctness_counts(path: str):
    """Return (n_correct, n_total) from the Stage 3 labels, or None if not generated yet."""
    p = Path(path)
    if not p.exists():
        return None
    labels = pd.read_csv(p, usecols=["is_correct"])["is_correct"]
    return int(labels.sum()), len(labels)


@st.cache_data(show_spinner=False)
def load_saliency(path: str, max_samples: int = 1500, max_examples: int = 200):
    """Reduce the (large) Stage 4 dataset to the summaries and examples the page displays."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return None

    flat = []
    mean_correct = []
    mean_incorrect = []
    for s in data[:max_samples]:
        w = s.get("saliency_weights") or []
        if not w:
            continue
        flat.extend(w)
        (mean_correct if s.get("is_correct") else mean_incorrect).append(float(np.mean(w)))

    weights = np.array(flat, dtype=np.float32)
    counts, edges = np.histogram(weights, bins=60) if len(weights) else (np.array([]), np.array([]))
    return {
        "n_tokens": len(weights),
        "mean": float(weights.mean()) if len(weights) else 0.0,
        "median": float(np.median(weights)) if len(weights) else 0.0,
        "min": float(weights.min()) if len(weights) else 0.0,
        "max": float(weights.max()) if len(weights) else 0.0,
        "hist": pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts}),
        "mean_correct": mean_correct,
        "mean_incorrect": mean_incorrect,
        "examples": [s for s in data[:max_examples] if s.get("saliency_weights")],
    }


def label(css_class: str, text: str):
    st.markdown(f'<div class="{css_class}">{text}</div>', unsafe_allow_html=True)


def style_chart(chart):
    """Recessive axes and grid so the marks carry the reading."""
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            grid=False,
            domainColor="#DCE5E3",
            tickColor="#DCE5E3",
            labelColor="#5B6F6B",
            titleColor="#5B6F6B",
            titleFontWeight="normal",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_axisY(grid=True, gridColor="#EDF3F1", domain=False, ticks=False)
        .configure_legend(orient="top", title=None, labelColor="#5B6F6B", labelFontSize=11)
    )


# =========================================================================
# Chat view
# =========================================================================
def clear_conversation():
    st.session_state.messages = []


def queue_question(question: str):
    st.session_state.pending_prompt = question


def render_empty_state():
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="brand-mark">{BRAND_ICON}</div>
            <h2>Bạn muốn hỏi điều gì về sức khỏe?</h2>
            <p>Mô tả triệu chứng hoặc thắc mắc của bạn, hoặc bắt đầu từ một câu hỏi gợi ý bên dưới.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="suggestions"):
        for i in range(0, len(SUGGESTED_QUESTIONS), 2):
            for col, question in zip(st.columns(2), SUGGESTED_QUESTIONS[i : i + 2]):
                col.button(
                    question,
                    key=f"suggest_{SUGGESTED_QUESTIONS.index(question)}",
                    on_click=queue_question,
                    args=(question,),
                    width="stretch",
                )


def stream_answer():
    client = OpenAI(api_key=OPENAI_API_KEY)
    with st.chat_message("assistant", avatar=":material/stethoscope:"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing-dots">'
            '<span class="dot"></span><span class="dot"></span><span class="dot"></span>'
            '<span class="label">Đang suy nghĩ…</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        full_response = ""
        try:
            for chunk in client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *st.session_state.messages],
                stream=True,
            ):
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if not delta:
                    continue
                full_response += delta
                placeholder.markdown(
                    full_response + '<span class="stream-caret">▌</span>',
                    unsafe_allow_html=True,
                )
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            placeholder.empty()
            st.error(f"Không thể lấy câu trả lời, vui lòng thử lại. Chi tiết: {e}", icon=":material/error:")


def render_chat():
    # Rendered at script root (not inside a container) so Streamlit pins it to the viewport bottom.
    prompt = st.chat_input("Hỏi tôi về vấn đề sức khỏe của bạn...")
    prompt = prompt or st.session_state.pop("pending_prompt", None)

    key_missing = bool(prompt) and not OPENAI_API_KEY
    if prompt and not key_missing:
        st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.messages:
        st.button("Hội thoại mới", key="clear_btn", icon=":material/add_comment:", on_click=clear_conversation)

    with st.container(key="chat_view"):
        if key_missing:
            st.error(
                "Chưa cấu hình `OPENAI_API_KEY`. Hãy thêm vào file `.env` rồi khởi động lại ứng dụng.",
                icon=":material/key:",
            )
        if not st.session_state.messages:
            render_empty_state()
            return

        for message in st.session_state.messages:
            avatar = None if message["role"] == "user" else ":material/stethoscope:"
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt and not key_missing:
            stream_answer()


# =========================================================================
# Pipeline view
# =========================================================================
def stage_header(index: int):
    stage = STAGES[index]
    label("stage-phase", html.escape(stage["phase"]))
    label("stage-title", html.escape(stage["title"]))
    st.markdown(stage["summary"])
    st.caption(
        f"**Script:** `{stage['script']}`  \n"
        f"**Inputs:** {stage['inputs']}  \n"
        f"**Outputs:** {stage['outputs']}"
    )


def correctness_section(counts):
    n_correct, n_total = counts
    n_incorrect = n_total - n_correct
    col1, col2, col3 = st.columns(3)
    col1.metric("Total samples", f"{n_total:,}", height="stretch")
    col2.metric(
        "Correct (≥0.7 cosine)", f"{n_correct:,}", f"{n_correct / n_total:.1%} of samples",
        delta_color="off", delta_arrow="off",
    )
    col3.metric(
        "Incorrect", f"{n_incorrect:,}", f"{n_incorrect / n_total:.1%} of samples",
        delta_color="off", delta_arrow="off",
    )


def saliency_charts(sal):
    col1, col2, col3 = st.columns(3)

    with col1:
        label("chart-title", "Saliency-weight distribution (post-tanh)")
        hist = (
            alt.Chart(sal["hist"])
            .mark_bar(color=ACCENT)
            .encode(
                x=alt.X("start:Q", title="Saliency weight"),
                x2="end:Q",
                y=alt.Y("count:Q", title="Token count", axis=alt.Axis(format="~s", tickCount=4)),
                y2=alt.datum(0),
                tooltip=[
                    alt.Tooltip("start:Q", title="from", format=".2f"),
                    alt.Tooltip("end:Q", title="to", format=".2f"),
                    alt.Tooltip("count:Q", title="tokens", format=","),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(style_chart(hist), width="stretch")

    with col2:
        label("chart-title", "Per-sample mean saliency, by correctness")
        if sal["mean_correct"] and sal["mean_incorrect"]:
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
                .mark_area(opacity=0.45, line=True)
                .encode(
                    x=alt.X("mean_weight:Q", title="Per-sample mean saliency"),
                    y=alt.Y("density:Q", title="Density", axis=alt.Axis(tickCount=4)),
                    color=alt.Color(
                        "label:N",
                        scale=alt.Scale(domain=["Correct", "Incorrect"], range=[ACCENT, COUNTERPOINT]),
                    ),
                    tooltip=["label:N", alt.Tooltip("mean_weight:Q", format=".2f")],
                )
                .properties(height=220)
            )
            st.altair_chart(style_chart(density), width="stretch")

    with col3:
        label("chart-title", "Tanh smoothing curve (raw → scaled weight)")
        st.altair_chart(style_chart(tanh_curve()), width="stretch")


def tanh_curve():
    raw = np.linspace(0.0, 3.0, 200)
    min_val, max_val, scale_factor = 0.2, 2.5, 1.0
    mean_v = (max_val + min_val) / 2
    amp = (max_val - min_val) / 2
    scaled = mean_v + amp * np.tanh((raw - 1.0) / scale_factor)
    curve = (
        alt.Chart(pd.DataFrame({"raw": raw, "scaled": scaled}))
        .mark_line(color=ACCENT, strokeWidth=2)
        .encode(
            x=alt.X("raw:Q", title="Raw weight (mean-normalized)"),
            y=alt.Y("scaled:Q", title="Scaled weight", scale=alt.Scale(domain=[0, 3]), axis=alt.Axis(tickCount=4)),
            tooltip=[alt.Tooltip("raw:Q", format=".2f"), alt.Tooltip("scaled:Q", format=".2f")],
        )
    )
    bounds = (
        alt.Chart(pd.DataFrame({"y": [min_val, max_val]}))
        .mark_rule(strokeDash=[4, 4], color=NEUTRAL)
        .encode(y="y:Q")
    )
    return (curve + bounds).properties(height=220)


def token_heatmap(examples):
    label("chart-title", "Example: token-level saliency heatmap")
    example_idx = st.slider(
        "Pick a sample", min_value=1, max_value=len(examples), value=1, key="saliency_example"
    )
    sample = examples[example_idx - 1]
    weights_ex = sample.get("saliency_weights") or []
    # Approximate per-word segments: saliency is per-token, the display is per-word.
    words = (sample.get("completion") or "").split()
    n = min(len(words), len(weights_ex))
    if n == 0:
        return

    w_arr = np.array(weights_ex, dtype=np.float32)
    idx = np.linspace(0, len(w_arr), num=n + 1, dtype=int)
    word_weights = np.array([float(w_arr[idx[i] : max(idx[i + 1], idx[i] + 1)].mean()) for i in range(n)])
    w_min, w_max = float(word_weights.min()), float(word_weights.max())

    def color_for(w: float) -> str:
        norm = 0.5 if w_max - w_min < 1e-6 else (w - w_min) / (w_max - w_min)
        # Interpolate #E6F2F0 → #0F766E, switching ink where white becomes more legible.
        r = int(230 + (15 - 230) * norm)
        g = int(242 + (118 - 242) * norm)
        b = int(240 + (110 - 240) * norm)
        ink = "#FFFFFF" if norm > 0.7 else "#17312D"
        return f"background:rgb({r},{g},{b});color:{ink};"

    spans = "".join(
        f'<span title="{w:.2f}" style="{color_for(w)}">{html.escape(word)}</span> '
        for word, w in zip(words[:n], word_weights)
    )
    st.markdown(
        f'<div class="token-map">{spans}</div>'
        f'<div class="token-legend">{w_min:.2f}<i></i>{w_max:.2f} '
        "&nbsp;darker = more salient (token contributes more to the loss)</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Question: *{sample.get('question', '')[:140]}…*  •  "
        f"is_correct=`{sample.get('is_correct')}`  •  "
        "word-level average of token weights (approximate alignment)."
    )


def results_section():
    label("chart-title", "Reported results (test set, 463 samples)")
    metric_order = ["BLEU-4", "ROUGE-1", "ROUGE-L", "BERTScore F1"]
    results_df = pd.DataFrame(
        {
            "Method": METHOD_ORDER,
            "BLEU-4": [0.28, 0.31, 0.35],
            "ROUGE-1": [0.51, 0.54, 0.58],
            "ROUGE-L": [0.42, 0.45, 0.49],
            "BERTScore F1": [0.81, 0.83, 0.86],
        }
    )
    delta_col1, delta_col2, delta_col3 = st.columns(3)
    delta_col1.metric("BLEU-4 (TVAFT vs SFT)", "0.35", "+25.0%")
    delta_col2.metric("ROUGE-L (TVAFT vs SFT)", "0.49", "+16.7%")
    delta_col3.metric("BERTScore F1 (TVAFT vs SFT)", "0.86", "+6.2%")

    metrics_long = results_df.melt(id_vars="Method", var_name="Metric", value_name="Score")
    metrics_chart = (
        alt.Chart(metrics_long)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "Metric:N", title=None, sort=metric_order,
                axis=alt.Axis(labelAngle=0), scale=alt.Scale(paddingInner=0.3),
            ),
            xOffset=alt.XOffset("Method:N", sort=METHOD_ORDER, scale=alt.Scale(paddingInner=0.08)),
            y=alt.Y("Score:Q", title="Score", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(tickCount=5)),
            color=alt.Color("Method:N", scale=alt.Scale(domain=METHOD_ORDER, range=METHOD_COLORS)),
            tooltip=["Method", "Metric", alt.Tooltip("Score:Q", format=".2f")],
        )
        .properties(height=260)
    )
    st.altair_chart(style_chart(metrics_chart), width="stretch")

    score_cols = metric_order
    st.dataframe(
        results_df.style.format({c: "{:.2f}" for c in score_cols}).highlight_max(
            subset=score_cols, color="#D5EAE6"
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Significance: TVAFT > SFT on all metrics (*p* < 0.01); "
        "TVAFT > ReFT on BLEU-4 and ROUGE-L (*p* < 0.05). "
        "Test split = 463 held-out samples, greedy decoding (`temperature=0.0`)."
    )


def comparison_section():
    label("section-title", "Comparison with Baselines")
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
    st.dataframe(comparison_df, hide_index=True, width="stretch")


def hyperparameters_section():
    label("section-title", "Key TVAFT Hyperparameters")
    col1, col2 = st.columns(2)
    with col1.container(border=True, height="stretch"):
        st.markdown(
            "- **τ_sim** = `0.70` — Sentence-BERT cosine threshold for correctness\n"
            "- **τ_correct** = `0.95` — bonus cutoff for confident correct tokens\n"
            "- **τ_incorrect** = `0.01` — minimum *p*(judge) for boosting incorrect samples\n"
            "- **τ_r** = `1.2` — sigmoid pivot for ratio-based saliency\n"
            "- **ε** = `1e-9` — numerical-stability constant"
        )
    with col2.container(border=True, height="stretch"):
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


def render_pipeline():
    st.markdown(
        """
        <div class="page-intro">
            <h2>Token Value-Aware Fine-Tuning (TVAFT)</h2>
            <p>End-to-end pipeline that turns a Vietnamese medical QA dataset into a
            saliency-weighted training set, then fine-tunes <strong>PhoGPT-4B-Chat</strong>
            with token-level importance signals.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipeline stages", "5")
    col2.metric("Base model", "PhoGPT-4B-Chat", help="vinai/PhoGPT-4B-Chat, 4-bit NF4 (QLoRA)")
    col3.metric("Train / Test", "4,167 / 463", help="hungnm/vietnamese-medical-qa, 10% test split")
    col4.metric("Method", "TVAFT + LoRA")

    label("section-title", "Pipeline Flow")
    st.graphviz_chart(FLOW_DOT, width="stretch")

    with st.spinner("Đang tải dữ liệu pipeline…"):
        counts = correctness_counts(str(BERT_LABELS_PATH))
        sal = load_saliency(str(TVAFT_DATASET_PATH))

    label("section-title", "Stages")
    with st.container(key="timeline"):
        with st.container(key="stage_0"):
            stage_header(0)
        with st.container(key="stage_1"):
            stage_header(1)
        with st.container(key="stage_2"):
            stage_header(2)
            if counts:
                correctness_section(counts)
            else:
                st.info("Run stage 3 to populate `03_bert_labels.csv` and see the correctness summary here.")
        with st.container(key="stage_3"):
            stage_header(3)
            if sal and sal["n_tokens"]:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Tokens scored", f"{sal['n_tokens']:,}", help="Across the first 1,500 samples.")
                col2.metric("Mean weight", f"{sal['mean']:.2f}")
                col3.metric("Median", f"{sal['median']:.2f}")
                col4.metric("Range", f"{sal['min']:.2f} – {sal['max']:.2f}")
                saliency_charts(sal)
                if sal["examples"]:
                    token_heatmap(sal["examples"])
            else:
                st.info("Run stage 4 to populate `tvaft_finetune_dataset.json` and see saliency charts here.")
        with st.container(key="stage_4"):
            stage_header(4)
            results_section()

    comparison_section()
    hyperparameters_section()


# =========================================================================
# Page
# =========================================================================
st.set_page_config(
    page_title="Vietnamese Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.html(APP_CSS)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    f"""
    <div class="app-header">
        <div class="brand-mark">{BRAND_ICON}</div>
        <div>
            <div class="brand-name">Vietnamese Medical Chatbot</div>
            <div class="brand-tagline">Cố vấn sức khỏe thông minh cho người Việt</div>
        </div>
    </div>
    <div class="medical-disclaimer">
        <span class="disclaimer-full"><strong>Tuyên bố miễn trừ trách nhiệm:</strong> Đây là hệ thống thử nghiệm sử dụng
        trí tuệ nhân tạo, câu trả lời chỉ mang tính tham khảo. Luôn tham khảo ý kiến bác sĩ chuyên môn; trường hợp khẩn cấp hãy gọi 115.</span>
        <span class="disclaimer-short"><strong>Chỉ mang tính tham khảo,</strong> không thay thế ý kiến bác sĩ. Trường hợp khẩn cấp hãy gọi 115.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

VIEWS = {"chat": ":material/chat: Chatbot", "pipeline": ":material/account_tree: TVAFT Pipeline"}
with st.container(key="nav"):
    view = st.segmented_control(
        "Điều hướng",
        options=list(VIEWS),
        format_func=VIEWS.get,
        default="chat",
        required=True,
        key="view",
        label_visibility="collapsed",
    )

if view == "pipeline":
    render_pipeline()
else:
    render_chat()

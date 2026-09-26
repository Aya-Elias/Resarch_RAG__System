"""
Smart Research Paper Discovery & QA Assistant
================================================
A single-page Streamlit app that lets a user search arXiv (or upload a PDF),
pick up to 2 papers, get an executive/comparison summary, and then chat with
the paper(s) via a grounded, citation-backed RAG pipeline.

Pipeline stages (see inline section markers below):
  A. Discovery      - arXiv search + query rewriting + semantic re-ranking
  B. Upload         - user-supplied PDF as an alternative/addition
  C. Clean          - download, extract (PyMuPDF), generic text cleaning
  D. Summarize      - executive summary (1 paper) / comparison (2 papers)
  E. Chunking       - structure-aware adaptive chunking
  F. Embeddings     - BAAI/bge-small-en-v1.5 (local, sentence-transformers)
  G. Vector store   - FAISS IndexFlatIP over L2-normalized vectors
  H. Retrieval      - FAISS top-15 -> cross-encoder rerank -> top-5
  I. Answer         - grounded QA with citations + sliding conversation memory
"""

import io
import os
import re
import time
import html
import requests
import numpy as np
import xml.etree.ElementTree as ET
from collections import Counter

import streamlit as st

# ============================================================================
# Page config & styling
# ============================================================================

st.set_page_config(
    page_title="Paper Discovery & QA Assistant",
    page_icon="📚",
    layout="wide",
)


def inject_css():
    st.markdown(
        """
        <style>
        /* ---- palette: calm academic ---- */
        :root {
            --navy: #1c2f4a;
            --navy-light: #2e4568;
            --paper-bg: #faf6ee;
            --card-bg: #ffffff;
            --muted-gray: #6b6b6b;
            --burgundy: #7a2e35;
            --border-soft: #e4ddcf;
        }

        .stApp {
            background-color: var(--paper-bg);
        }

        html, body, [class*="css"] {
            font-family: "Source Sans Pro", "Helvetica Neue", sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: "Georgia", "Iowan Old Style", serif !important;
            color: var(--navy) !important;
        }

        /* header */
        .app-header {
            padding: 0.25rem 0 0.75rem 0;
            border-bottom: 2px solid var(--navy);
            margin-bottom: 1.25rem;
        }
        .app-header p {
            color: var(--muted-gray);
            font-size: 0.95rem;
            margin-top: -0.4rem;
        }

        /* result card */
        .paper-card {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
        }
        .paper-card.selected {
            border: 2px solid var(--navy);
            background: #f2f6fb;
        }
        .paper-title {
            font-weight: 600;
            color: var(--navy);
            font-size: 1.02rem;
            margin-bottom: 0.15rem;
        }
        .paper-meta {
            color: var(--muted-gray);
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }
        .paper-score {
            display: inline-block;
            background: var(--navy);
            color: white;
            font-size: 0.72rem;
            padding: 0.05rem 0.5rem;
            border-radius: 10px;
            margin-left: 0.4rem;
        }
        .paper-abstract {
            color: #333;
            font-size: 0.88rem;
            line-height: 1.4;
        }

        /* summary card */
        .summary-card {
            background: var(--card-bg);
            border-left: 4px solid var(--burgundy);
            border-radius: 4px;
            padding: 1rem 1.25rem;
            margin: 1rem 0 1.5rem 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .summary-label {
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-size: 0.72rem;
            color: var(--burgundy);
            font-weight: 700;
            margin-bottom: 0.4rem;
        }

        /* chat bubbles */
        .chat-row { margin-bottom: 1rem; }
        .bubble-q {
            background: var(--navy);
            color: white;
            padding: 0.6rem 0.9rem;
            border-radius: 12px 12px 2px 12px;
            max-width: 80%;
            margin-left: auto;
            font-size: 0.92rem;
        }
        .bubble-a {
            background: var(--card-bg);
            border: 1px solid var(--border-soft);
            color: #222;
            padding: 0.7rem 0.95rem;
            border-radius: 12px 12px 12px 2px;
            max-width: 85%;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .bubble-sources {
            color: var(--muted-gray);
            font-size: 0.76rem;
            margin-top: 0.35rem;
            font-style: italic;
        }
        .selection-badge {
            display: inline-block;
            background: var(--navy-light);
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 14px;
            font-size: 0.85rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Secrets & cached model / client loaders
# ============================================================================

def get_groq_api_key():
    key = st.secrets.get("GROQ_API_KEY")
    if not key:
        st.error(
            "Missing `GROQ_API_KEY` in Streamlit secrets. Add it under "
            "**Settings -> Secrets** before using this app."
        )
        st.stop()
    return key


def get_gemini_api_key():
    # Optional - image understanding degrades gracefully if absent.
    return st.secrets.get("GEMINI_API_KEY")


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("BAAI/bge-reranker-base")


@st.cache_resource(show_spinner=False)
def get_groq_client(api_key):
    from groq import Groq
    return Groq(api_key=api_key)


@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key):
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================================
# Bilingual UI support (Arabic + English) - UI/chat-language layer only.
# Does not touch the retrieval pipeline (embeddings, chunking, reranker).
# ============================================================================

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")


def contains_arabic(text: str) -> bool:
    return bool(ARABIC_CHAR_RE.search(text or ""))


UI_TEXT = {
    "en": {
        "app_title": "📚 Smart Research Paper Discovery & QA Assistant",
        "app_subtitle": "Search arXiv or upload a PDF, pick up to two papers, and get answers "
                         "grounded only in what they actually say - with page citations.",
        "lang_label": "Language",
        "sidebar_session": "### Session",
        "sidebar_loaded": "Loaded paper(s):",
        "sidebar_none_loaded": "No paper(s) loaded yet.",
        "sidebar_choose_different": "↩ Choose different paper(s)",
        "sidebar_clear": "🗑 Clear / start over",
        "section1": "#### 1. Search arXiv or upload a PDF",
        "search_placeholder": "e.g. How do transformers detect anomalies in time series data?",
        "search_btn": "🔍 Search arXiv",
        "empty_topic_warning": "Enter a topic or question first.",
        "spinner_search": "Rewriting query and searching arXiv...",
        "arxiv_search_failed": "arXiv search failed: {e}",
        "searched_for": "Searched arXiv for: *{query}*",
        "translated_from": "Searched arXiv for: *{query}* (translated from your original: *{original}*)",
        "upload_expander": "📄 Or upload your own PDF",
        "upload_label": "Upload a PDF",
        "upload_include": "Include \u201c{name}\u201d in your paper selection",
        "upload_limit_caption": "You already have 2 papers selected - uncheck one to add this instead.",
        "section2": "#### 2. Select up to 2 papers",
        "selected_badge": "{count} of 2 papers selected",
        "select_this_paper": "Select this paper",
        "process_btn": "🚀 Process {count} selected paper(s)",
        "too_many_selected": "Please select at most 2 papers.",
        "spinner_download": "Downloading, extracting, and cleaning paper text...",
        "spinner_summary": "Generating summary...",
        "spinner_kb": "Building the knowledge base (chunking + embeddings)...",
        "processing_failed": "Something went wrong while processing the paper(s): {e}",
        "section3": "#### 3. Summary",
        "summary_executive": "Executive Summary",
        "summary_comparison": "Comparison Summary",
        "section4": "#### 4. Ask questions about the paper(s)",
        "chat_placeholder": "Ask a question about the selected paper(s)...",
        "spinner_answer": "Retrieving relevant passages and answering...",
        "answer_failed": "Couldn't answer that question: {e}",
        "sources_label": "Sources",
        "sources_none": "none",
    },
    "ar": {
        "app_title": "📚 مساعد اكتشاف الأبحاث والإجابة عن أسئلتها",
        "app_subtitle": "ابحثي في arXiv أو ارفعي ملف PDF، اختاري حتى ورقتين، واحصلي على إجابات "
                         "مبنية فقط على محتوى الورقة نفسها - مع ذكر رقم الصفحة.",
        "lang_label": "اللغة",
        "sidebar_session": "### الجلسة",
        "sidebar_loaded": "الأوراق المحمّلة:",
        "sidebar_none_loaded": "لا توجد أوراق محمّلة بعد.",
        "sidebar_choose_different": "↩ اختيار أوراق مختلفة",
        "sidebar_clear": "🗑 مسح / البدء من جديد",
        "section1": "#### ١. ابحثي في arXiv أو ارفعي ملف PDF",
        "search_placeholder": "مثال: إزاي الـ transformers بتكتشف الشذوذ في بيانات السلاسل الزمنية؟",
        "search_btn": "🔍 ابحث في arXiv",
        "empty_topic_warning": "اكتبي موضوع أو سؤال البحث أولاً.",
        "spinner_search": "جارٍ إعادة صياغة البحث والبحث في arXiv...",
        "arxiv_search_failed": "فشل البحث في arXiv: {e}",
        "searched_for": "تم البحث في arXiv عن: *{query}*",
        "translated_from": "تم البحث في arXiv عن: *{query}* (مترجم من سؤالك الأصلي: *{original}*)",
        "upload_expander": "📄 أو ارفعي ملف PDF الخاص بك",
        "upload_label": "ارفعي ملف PDF",
        "upload_include": "تضمين \u201c{name}\u201d في اختيارك",
        "upload_limit_caption": "عندك بالفعل ورقتين مختارتين - شيلي واحدة عشان تضيفي دي.",
        "section2": "#### ٢. اختاري حتى ورقتين",
        "selected_badge": "تم اختيار {count} من 2",
        "select_this_paper": "اختيار هذه الورقة",
        "process_btn": "🚀 معالجة {count} ورقة مختارة",
        "too_many_selected": "من فضلك اختاري ورقتين بحد أقصى.",
        "spinner_download": "جارٍ تنزيل الورقة واستخراج النص وتنظيفه...",
        "spinner_summary": "جارٍ إنشاء الملخص...",
        "spinner_kb": "جارٍ بناء قاعدة المعرفة (تقطيع النص + التمثيلات الرقمية)...",
        "processing_failed": "حصلت مشكلة أثناء معالجة الورقة/الأوراق: {e}",
        "section3": "#### ٣. الملخص",
        "summary_executive": "ملخص تنفيذي",
        "summary_comparison": "ملخص مقارن",
        "section4": "#### ٤. اسألي عن الورقة/الأوراق",
        "chat_placeholder": "اكتبي سؤالك عن الورقة/الأوراق المختارة...",
        "spinner_answer": "جارٍ البحث عن المقاطع المناسبة والإجابة...",
        "answer_failed": "معرفتش أجاوب على السؤال ده: {e}",
        "sources_label": "المصادر",
        "sources_none": "لا يوجد",
    },
}


def get_lang() -> str:
    return st.session_state.get("lang", "en")


def t(key: str, **kwargs) -> str:
    """Look up a UI string in the active language and format it."""
    text = UI_TEXT[get_lang()].get(key, UI_TEXT["en"].get(key, key))
    return text.format(**kwargs) if kwargs else text


def inject_rtl_overrides():
    """Only injected when Arabic is active. Flips direction/alignment on our
    own text containers (labels, summary card, chat bubbles) to RTL, while
    keeping paper cards (always English metadata) explicitly LTR."""
    st.markdown(
        """
        <style>
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .stAlert, .stAlert p,
        .stButton button p,
        [data-testid="stTextInput"] label,
        [data-testid="stFileUploaderDropzone"] div,
        .stCheckbox label, .stRadio label,
        .selection-badge, .bubble-q, .bubble-a, .bubble-sources,
        .summary-card, .summary-label, .app-header, .app-header p,
        h1, h2, h3, h4 {
            direction: rtl !important;
            text-align: right !important;
        }
        .paper-card, .paper-title, .paper-meta, .paper-score,
        .paper-abstract, .keep-ltr {
            direction: ltr !important;
            text-align: left !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Stage A - Discovery: arXiv search + query rewriting + semantic re-ranking
# ============================================================================

def rewrite_query(groq_client, user_question: str) -> str:
    """Turn a natural-language question into arXiv-friendly keywords."""
    prompt = f"""Extract 3-6 key academic search terms from this question.
Return ONLY the terms separated by spaces, no explanation.

Question: {user_question}
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def translate_to_english(groq_client, text: str) -> str:
    """arXiv's API and content are English-only, so an Arabic query is
    translated before it ever reaches the search/keyword-extraction steps."""
    prompt = f"""Translate the following text to English. Return ONLY the translation,
nothing else.

Text: {text}
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def search_arxiv(query: str, max_results: int = 20):
    """Search arXiv for papers matching the query (keyword match, not semantic)."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ns).text.strip()
        published = entry.find("atom:published", ns).text[:10]
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib["href"]

        papers.append(
            {
                "title": title,
                "authors": authors,
                "published": published,
                "summary": summary,
                "pdf_url": pdf_url,
            }
        )
    return papers


def embed_text(model, text: str):
    """Local embedding, no API call."""
    return model.encode(text)


def embed_texts(model, texts: list):
    """Batch local embedding - much faster than encoding one text at a time."""
    if not texts:
        return np.array([])
    return model.encode(texts, show_progress_bar=False, batch_size=32)


def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def semantic_search_papers(groq_client, embedding_model, user_question: str, max_results: int = 20):
    """
    Meaning-aware search: (if needed) translate the question to English -> rewrite
    it -> fetch a candidate pool from arXiv -> re-rank by semantic similarity to
    the (English) question, since arXiv only does keyword matching and its
    content is English-only.
    """
    was_translated = contains_arabic(user_question)
    english_question = translate_to_english(groq_client, user_question) if was_translated else user_question

    search_query = rewrite_query(groq_client, english_question)
    candidates = search_arxiv(search_query, max_results=max_results)
    question_embedding = embed_text(embedding_model, english_question)

    if candidates:
        paper_embeddings = embed_texts(embedding_model, [p["summary"] for p in candidates])
        for paper, paper_embedding in zip(candidates, paper_embeddings):
            paper["relevance_score"] = cosine_similarity(question_embedding, paper_embedding)

    ranked = sorted(candidates, key=lambda p: p["relevance_score"], reverse=True)
    return ranked, search_query, user_question, was_translated


# ============================================================================
# Stage C - Download, Extract & Clean
# ============================================================================

def download_pdf(pdf_url: str, save_dir: str = "papers") -> str:
    os.makedirs(save_dir, exist_ok=True)
    filename = pdf_url.split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    local_path = os.path.join(save_dir, filename)

    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def extract_text_from_pdf(pdf_path: str):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc, start=1):
        pages.append({"page_number": page_num, "text": page.get_text()})
    doc.close()
    return pages


def _fix_hyphenation(text: str) -> str:
    """'trans-\\nformer' -> 'transformer'."""
    return re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_page_number_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(re.fullmatch(r"(page\s*)?\d{1,4}(\s*of\s*\d{1,4})?", stripped, re.IGNORECASE))


def _detect_repeated_lines(pages, min_repeat_ratio: float = 0.4) -> set:
    """Repeated first/last lines across pages -> running headers/footers."""
    first_lines, last_lines = [], []
    for page in pages:
        lines = [l.strip() for l in page["text"].split("\n") if l.strip()]
        if lines:
            first_lines.append(lines[0])
            last_lines.append(lines[-1])

    total_pages = len(pages)
    threshold = max(2, int(total_pages * min_repeat_ratio))
    return {line for line, count in Counter(first_lines + last_lines).items() if count >= threshold}


def clean_pages(pages):
    """Generic cleaning: strip repeated headers/footers, page numbers, fix
    hyphenation, normalize whitespace. Works on any paper's layout."""
    repeated_lines = _detect_repeated_lines(pages)
    cleaned_pages = []

    for page in pages:
        lines = page["text"].split("\n")
        kept_lines = [l for l in lines if l.strip() not in repeated_lines and not _is_page_number_line(l)]
        cleaned_text = "\n".join(kept_lines)
        cleaned_text = _fix_hyphenation(cleaned_text)
        cleaned_text = _normalize_whitespace(cleaned_text)
        cleaned_pages.append({"page_number": page["page_number"], "text": cleaned_text})

    return cleaned_pages


def process_arxiv_paper(paper: dict) -> dict:
    """Download -> extract -> clean for an arXiv search result."""
    local_path = download_pdf(paper["pdf_url"])
    pages = extract_text_from_pdf(local_path)
    cleaned = clean_pages(pages)
    return {
        "title": paper["title"],
        "authors": paper.get("authors", []),
        "published": paper.get("published", ""),
        "local_path": local_path,
        "cleaned_pages": cleaned,
        "source": "arxiv",
    }


def process_uploaded_paper(file_bytes: bytes, filename: str) -> dict:
    """Save -> extract -> clean for a user-uploaded PDF."""
    os.makedirs("papers", exist_ok=True)
    local_path = os.path.join("papers", filename)
    with open(local_path, "wb") as f:
        f.write(file_bytes)

    pages = extract_text_from_pdf(local_path)
    cleaned = clean_pages(pages)
    return {
        "title": filename,
        "authors": [],
        "published": "",
        "local_path": local_path,
        "cleaned_pages": cleaned,
        "source": "upload",
    }


# ============================================================================
# Stage D - Summarization
# ============================================================================

def generate_executive_summary(groq_client, cleaned_pages) -> str:
    sample_text = "\n\n".join(p["text"] for p in cleaned_pages[:4])
    prompt = f"""Based on this excerpt from an academic paper, write a short executive summary
covering: (1) the paper's goal/problem, (2) the methodology used, (3) key findings/results.
Keep it to 4-6 sentences total.

Paper excerpt:
{sample_text[:6000]}
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def generate_comparison_summary(groq_client, paper_a: dict, paper_b: dict) -> str:
    sample_a = "\n\n".join(p["text"] for p in paper_a["cleaned_pages"][:4])[:6000]
    sample_b = "\n\n".join(p["text"] for p in paper_b["cleaned_pages"][:4])[:6000]

    prompt = f"""You are comparing two academic papers based on excerpts from each. Write a
comparison summary covering: (1) their shared focus/topic, (2) how their methodologies differ,
(3) how their findings/results differ. Be explicit about which paper does what by name.
Keep it to 5-8 sentences total.

Paper A: "{paper_a['title']}"
Excerpt A:
{sample_a}

Paper B: "{paper_b['title']}"
Excerpt B:
{sample_b}
"""
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ============================================================================
# Stage E - Adaptive Chunking (structure-aware, not fixed-size / not semantic)
# ============================================================================

def split_into_paragraphs(text: str):
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str):
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def adaptive_chunk_page(
    page_text: str,
    page_number: int,
    paper_title: str,
    min_chunk_chars: int = 300,
    max_chunk_chars: int = 1400,
    overlap_chars: int = 150,
):
    """
    Structure-aware adaptive chunking for one page:
    - paragraph boundaries first
    - oversized paragraphs split on sentence boundaries
    - undersized chunks merged forward
    - small overlap added between consecutive chunks for continuity
    Defaults tuned for bge-small-en-v1.5's 512-token sequence length.
    """
    paragraphs = split_into_paragraphs(page_text)
    raw_chunks = []
    buffer = ""

    for para in paragraphs:
        if len(para) > max_chunk_chars:
            sentences = split_into_sentences(para)
            sentence_buffer = ""
            for sentence in sentences:
                if len(sentence_buffer) + len(sentence) <= max_chunk_chars:
                    sentence_buffer += (" " if sentence_buffer else "") + sentence
                else:
                    if sentence_buffer:
                        raw_chunks.append(sentence_buffer)
                    sentence_buffer = sentence
            if sentence_buffer:
                raw_chunks.append(sentence_buffer)
        else:
            if len(buffer) + len(para) <= max_chunk_chars:
                buffer += ("\n\n" if buffer else "") + para
            else:
                if buffer:
                    raw_chunks.append(buffer)
                buffer = para

    if buffer:
        raw_chunks.append(buffer)

    # Merge chunks that are still too small into the next one.
    merged_chunks = []
    carry = ""
    for chunk in raw_chunks:
        candidate = (carry + "\n\n" + chunk) if carry else chunk
        if len(candidate) < min_chunk_chars:
            carry = candidate
        else:
            merged_chunks.append(candidate)
            carry = ""
    if carry:
        if merged_chunks:
            merged_chunks[-1] += "\n\n" + carry
        else:
            merged_chunks.append(carry)

    # Add overlap between consecutive chunks.
    final_chunks = []
    for i, chunk in enumerate(merged_chunks):
        if i > 0 and overlap_chars > 0:
            prev_tail = merged_chunks[i - 1][-overlap_chars:]
            chunk = prev_tail + " ... " + chunk
        final_chunks.append(
            {
                "page_number": page_number,
                "type": "text",
                "paper_title": paper_title,
                "text": chunk,
            }
        )
    return final_chunks


def chunk_cleaned_pages(cleaned_pages, paper_title: str):
    all_chunks = []
    for page in cleaned_pages:
        all_chunks.extend(adaptive_chunk_page(page["text"], page["page_number"], paper_title))
    return all_chunks


# ============================================================================
# Optional - Table extraction
# ============================================================================

def extract_tables_from_pdf(pdf_path: str, paper_title: str):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    table_chunks = []
    try:
        for page_num, page in enumerate(doc, start=1):
            tabs = page.find_tables()
            for table in tabs.tables:
                rows = table.extract()
                lines = [f"[Table from page {page_num}]"]
                for row in rows:
                    clean_row = [cell.strip() if cell else "" for cell in row]
                    lines.append(" | ".join(clean_row))
                table_chunks.append(
                    {
                        "page_number": page_num,
                        "type": "table",
                        "paper_title": paper_title,
                        "text": "\n".join(lines),
                    }
                )
    finally:
        doc.close()
    return table_chunks


# ============================================================================
# Optional - Image understanding via Gemini
# ============================================================================

def extract_images_from_pdf(pdf_path: str):
    import pymupdf
    doc = pymupdf.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({"page_number": page_num, "image_bytes": base_image["image"]})
    doc.close()
    return images


def describe_image(gemini_client, image_bytes: bytes, max_retries: int = 3):
    from PIL import Image
    image = Image.open(io.BytesIO(image_bytes))

    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    "Describe this figure from an academic paper in 2-4 sentences. "
                    "If it's a diagram, explain what it represents. "
                    "If it's a chart/plot, describe what's being compared.",
                    image,
                ],
            )
            return response.text.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise


def process_images(gemini_client, images, paper_title: str):
    described = []
    for img in images:
        try:
            description = describe_image(gemini_client, img["image_bytes"])
            described.append(
                {
                    "page_number": img["page_number"],
                    "type": "figure",
                    "paper_title": paper_title,
                    "text": f"[Figure from page {img['page_number']}]: {description}",
                }
            )
        except Exception:
            continue  # skip silently, never let one bad image crash the pipeline
    return described


# ============================================================================
# Stage F/G - Embeddings + FAISS Vector Store (combined across papers)
# ============================================================================

class PaperVectorStore:
    """FAISS-backed vector store over combined chunks from 1-2 papers.
    Every chunk keeps page_number + paper_title so results are attributable."""

    def __init__(self, chunks, embedding_model):
        import faiss

        self.chunks = chunks
        self.embedding_model = embedding_model

        vectors = np.array(embed_texts(embedding_model, [c["text"] for c in chunks]), dtype="float32")
        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

    def search(self, query: str, top_k: int = 15):
        import faiss

        query_vector = np.array([embed_text(self.embedding_model, query)], dtype="float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results


def build_combined_vector_store(paper_contexts, embedding_model, gemini_client):
    """Builds one vector store spanning all selected papers' text/table/figure chunks."""
    all_chunks = []
    for ctx in paper_contexts:
        text_chunks = chunk_cleaned_pages(ctx["cleaned_pages"], ctx["title"])
        all_chunks.extend(text_chunks)

        try:
            all_chunks.extend(extract_tables_from_pdf(ctx["local_path"], ctx["title"]))
        except Exception:
            pass  # optional feature - never block the core flow

        if gemini_client is not None:
            try:
                images = extract_images_from_pdf(ctx["local_path"])
                all_chunks.extend(process_images(gemini_client, images, ctx["title"]))
            except Exception:
                pass

    return PaperVectorStore(all_chunks, embedding_model)


# ============================================================================
# Stage H - Two-stage retrieval: FAISS (top 15) -> cross-encoder rerank (top 5)
# ============================================================================

def rerank_chunks(reranker_model, question: str, chunks, top_k: int = 5):
    if not chunks:
        return chunks
    pairs = [[question, chunk["text"]] for chunk in chunks]
    rerank_scores = reranker_model.predict(pairs)
    for chunk, score in zip(chunks, rerank_scores):
        chunk["rerank_score"] = float(score)
    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


# ============================================================================
# Stage I - Answer generation (grounded QA with citations + memory)
# ============================================================================

QA_SYSTEM_PROMPT = """You are answering questions about one or two specific academic papers, using
ONLY the context provided by the user in each turn. Follow these rules strictly:

1. Answer only from the given context. Do not use outside knowledge.
2. Do not infer or extrapolate beyond what is explicitly stated.
3. If the paper(s) don't address the question, reply with EXACTLY this sentence and nothing else,
   in the same language as the question:
   - English: "The paper(s) provided don't address this question."
   - Arabic: "الأوراق المقدمة لا تتناول هذا السؤال."
   Do not guess or add unrelated details.
4. Always cite the paper title and page number(s) your answer comes from, e.g. ("Paper title", page 5).
5. For specific numbers, metrics, or exact terms, quote them precisely as written in the context,
   but keep any direct quotation under 15 words - paraphrase everything else.
6. If the two papers conflict or differ on a point, state both versions explicitly and note which
   paper each one is from.
7. State your confidence: if the answer is fully supported, answer normally; if it's only partially
   supported or ambiguous, say so explicitly.
8. Keep answers concise (2-5 sentences) unless the question genuinely needs more detail.
9. Never say "based on the context" or "the provided text" - answer as if you've read the paper(s)
   directly.
10. Use the recent conversation turns only to resolve references in follow-up questions (e.g. "what
    about its size?") - never as a source of facts.
11. Detect the language of the user's question and respond in that same language. If you quote a
    short exact phrase from the paper (under 15 words), keep that quoted fragment in its original
    English, but write the surrounding explanation in the question's language.
"""


NO_ANSWER_PHRASES = (
    "the paper(s) provided don't address this question.",
    "الأوراق المقدمة لا تتناول هذا السؤال",
)


def is_no_answer(answer_text: str) -> bool:
    """True when the model's reply is the canonical 'no match found' refusal (in
    either language) - in that case retrieved chunks were irrelevant and
    shouldn't be shown as sources."""
    normalized = answer_text.strip().lower()
    return any(phrase.lower() in normalized for phrase in NO_ANSWER_PHRASES)


def build_qa_context(retrieved_chunks) -> str:
    parts = []
    for chunk in retrieved_chunks:
        label = f'[Source: "{chunk["paper_title"]}", page {chunk["page_number"]}]'
        parts.append(f"{label}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def answer_question(groq_client, vector_store, reranker_model, question: str, conversation_history, top_k: int = 5):
    # Stage 1: fast/approximate retrieval (bi-encoder, FAISS)
    retrieved = vector_store.search(question, top_k=15)
    # Stage 2: precise re-ranking (cross-encoder, true question<->chunk attention)
    retrieved = rerank_chunks(reranker_model, question, retrieved, top_k=top_k)

    context = build_qa_context(retrieved)

    messages = [{"role": "system", "content": QA_SYSTEM_PROMPT}]

    # Sliding window: last 3 Q&A turns, for resolving references only.
    for turn in conversation_history[-3:]:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})

    user_prompt = f"""Context:
{context}

Question: {question}"""
    messages.append({"role": "user", "content": user_prompt})

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=messages,
    )

    sources = sorted({(c["paper_title"], c["page_number"]) for c in retrieved}, key=lambda s: (s[0], s[1]))

    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": sources,
    }


# ============================================================================
# Session state management
# ============================================================================

def init_session_state():
    defaults = {
        "lang": "en",
        "search_results": [],
        "rewritten_query": "",
        "original_query": "",
        "query_was_translated": False,
        "selected_uploaded_file": None,  # (filename, bytes) staged for selection
        "include_upload_in_selection": False,
        "paper_contexts": [],
        "summary": "",
        "summary_kind": "",  # "executive" | "comparison"
        "vector_store": None,
        "chat_history": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def clear_all():
    for key in list(st.session_state.keys()):
        if key.startswith("chk_") or key in (
            "search_results",
            "rewritten_query",
            "original_query",
            "query_was_translated",
            "selected_uploaded_file",
            "include_upload_in_selection",
            "paper_contexts",
            "summary",
            "summary_kind",
            "vector_store",
            "chat_history",
            "topic_input",
        ):
            del st.session_state[key]
    init_session_state()


def choose_different_papers():
    """Keep search_results (no re-search) but clear the built paper session."""
    for key in list(st.session_state.keys()):
        if key.startswith("chk_"):
            del st.session_state[key]
    st.session_state.selected_uploaded_file = None
    st.session_state.include_upload_in_selection = False
    st.session_state.paper_contexts = []
    st.session_state.summary = ""
    st.session_state.summary_kind = ""
    st.session_state.vector_store = None
    st.session_state.chat_history = []


# ============================================================================
# UI sections
# ============================================================================

def selected_count() -> int:
    n = sum(1 for i in range(len(st.session_state.search_results)) if st.session_state.get(f"chk_{i}", False))
    if st.session_state.include_upload_in_selection and st.session_state.selected_uploaded_file:
        n += 1
    return n


def render_header():
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{t('app_title')}</h1>
            <p>{t('app_subtitle')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        lang_options = {"English": "en", "العربية": "ar"}
        labels = list(lang_options.keys())
        current_label = "English" if get_lang() == "en" else "العربية"
        choice = st.radio(t("lang_label"), labels, index=labels.index(current_label), key="lang_radio")
        st.session_state.lang = lang_options[choice]

        st.markdown(t("sidebar_session"))
        if st.session_state.paper_contexts:
            st.caption(t("sidebar_loaded"))
            for ctx in st.session_state.paper_contexts:
                st.write(f"- {ctx['title']}")
        else:
            st.caption(t("sidebar_none_loaded"))

        if st.session_state.paper_contexts:
            if st.button(t("sidebar_choose_different"), use_container_width=True):
                choose_different_papers()
                st.rerun()

        if st.button(t("sidebar_clear"), use_container_width=True):
            clear_all()
            st.rerun()


def render_search_and_upload(groq_client, embedding_model):
    st.markdown(t("section1"))
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "Research topic or question",
            key="topic_input",
            placeholder=t("search_placeholder"),
            label_visibility="collapsed",
        )
    with col2:
        search_clicked = st.button(t("search_btn"), use_container_width=True)

    if search_clicked:
        if not topic.strip():
            st.warning(t("empty_topic_warning"))
        else:
            with st.spinner(t("spinner_search")):
                try:
                    ranked, rewritten, original, was_translated = semantic_search_papers(
                        groq_client, embedding_model, topic.strip()
                    )
                    st.session_state.search_results = ranked
                    st.session_state.rewritten_query = rewritten
                    st.session_state.original_query = original
                    st.session_state.query_was_translated = was_translated
                    for key in list(st.session_state.keys()):
                        if key.startswith("chk_"):
                            del st.session_state[key]
                except Exception as e:
                    st.error(t("arxiv_search_failed", e=e))

    if st.session_state.rewritten_query:
        if st.session_state.query_was_translated:
            st.caption(
                t("translated_from", query=st.session_state.rewritten_query, original=st.session_state.original_query)
            )
        else:
            st.caption(t("searched_for", query=st.session_state.rewritten_query))

    with st.expander(t("upload_expander"), expanded=False):
        uploaded = st.file_uploader(t("upload_label"), type=["pdf"], label_visibility="collapsed")
        if uploaded is not None:
            st.session_state.selected_uploaded_file = (uploaded.name, uploaded.getvalue())
            already_in = st.session_state.include_upload_in_selection
            can_add = already_in or selected_count() < 2
            include = st.checkbox(
                t("upload_include", name=uploaded.name),
                value=already_in,
                disabled=not can_add,
                key="include_upload_checkbox",
            )
            st.session_state.include_upload_in_selection = include
            if not can_add and not already_in:
                st.caption(t("upload_limit_caption"))


def render_results(groq_client, embedding_model, gemini_client):
    if not st.session_state.search_results:
        return

    st.markdown(t("section2"))
    count = selected_count()
    badge_col, button_col = st.columns([3, 2])
    with badge_col:
        st.markdown(f'<span class="selection-badge">{t("selected_badge", count=count)}</span>', unsafe_allow_html=True)
    with button_col:
        if count > 0:
            render_process_button(groq_client, embedding_model, gemini_client, key_suffix="top")

    for i, paper in enumerate(st.session_state.search_results):
        key = f"chk_{i}"
        currently_checked = st.session_state.get(key, False)
        disabled = (not currently_checked) and (selected_count() >= 2)

        card_class = "paper-card selected" if currently_checked else "paper-card"
        authors = ", ".join(paper["authors"][:3]) + (" et al." if len(paper["authors"]) > 3 else "")
        abstract = paper["summary"][:320].replace("\n", " ")

        # Paper metadata (title/authors/abstract) is always English content,
        # so this card is intentionally left LTR regardless of UI language.
        st.markdown(
            f"""
            <div class="{card_class}">
                <div class="paper-title">{html.escape(paper['title'])}
                    <span class="paper-score">score {paper['relevance_score']:.2f}</span>
                </div>
                <div class="paper-meta">{html.escape(authors)} &middot; {paper['published']}</div>
                <div class="paper-abstract">{html.escape(abstract)}...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.checkbox(t("select_this_paper"), key=key, disabled=disabled, label_visibility="collapsed")


def render_process_button(groq_client, embedding_model, gemini_client, key_suffix="bottom"):
    count = selected_count()
    if count == 0:
        return

    if key_suffix == "bottom":
        st.markdown("---")
    process_clicked = st.button(
        t("process_btn", count=count),
        type="primary",
        key=f"process_btn_{key_suffix}",
        use_container_width=(key_suffix == "top"),
    )
    if not process_clicked:
        return

    selections = []
    for i, paper in enumerate(st.session_state.search_results):
        if st.session_state.get(f"chk_{i}", False):
            selections.append(("arxiv", paper))
    if st.session_state.include_upload_in_selection and st.session_state.selected_uploaded_file:
        selections.append(("upload", st.session_state.selected_uploaded_file))

    if len(selections) > 2:
        st.error(t("too_many_selected"))
        return

    paper_contexts = []
    try:
        with st.spinner(t("spinner_download")):
            for kind, item in selections:
                if kind == "arxiv":
                    paper_contexts.append(process_arxiv_paper(item))
                else:
                    filename, file_bytes = item
                    paper_contexts.append(process_uploaded_paper(file_bytes, filename))

        with st.spinner(t("spinner_summary")):
            if len(paper_contexts) == 1:
                summary = generate_executive_summary(groq_client, paper_contexts[0]["cleaned_pages"])
                summary_kind = "executive"
            else:
                summary = generate_comparison_summary(groq_client, paper_contexts[0], paper_contexts[1])
                summary_kind = "comparison"

        with st.spinner(t("spinner_kb")):
            vector_store = build_combined_vector_store(paper_contexts, embedding_model, gemini_client)

        st.session_state.paper_contexts = paper_contexts
        st.session_state.summary = summary
        st.session_state.summary_kind = summary_kind
        st.session_state.vector_store = vector_store
        st.session_state.chat_history = []
        st.rerun()

    except Exception as e:
        st.error(t("processing_failed", e=e))


def render_summary():
    if not st.session_state.summary:
        return

    label = t("summary_executive") if st.session_state.summary_kind == "executive" else t("summary_comparison")
    st.markdown(t("section3"))
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">{label}</div>
            <div>{html.escape(st.session_state.summary).replace(chr(10), '<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat(groq_client, reranker_model):
    if st.session_state.vector_store is None:
        return

    st.markdown(t("section4"))

    for turn in st.session_state.chat_history:
        st.markdown(f'<div class="chat-row"><div class="bubble-q">{html.escape(turn["question"])}</div></div>', unsafe_allow_html=True)

        no_match = is_no_answer(turn["answer"])
        if no_match:
            sources_line = ""
        else:
            # Paper titles are always English; wrap them LTR so they don't get
            # mirrored inside an RTL (Arabic) sources line.
            source_bits = [f'<span class="keep-ltr">"{html.escape(title)}", page {page}</span>' for title, page in turn["sources"]]
            sources_str = "; ".join(source_bits) if source_bits else t("sources_none")
            sources_line = f'<div class="bubble-sources">{t("sources_label")}: {sources_str}</div>'

        st.markdown(
            f"""
            <div class="chat-row">
                <div class="bubble-a">{html.escape(turn["answer"]).replace(chr(10), '<br>')}
                {sources_line}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    question = st.chat_input(t("chat_placeholder"))
    if question:
        with st.spinner(t("spinner_answer")):
            try:
                result = answer_question(
                    groq_client,
                    st.session_state.vector_store,
                    reranker_model,
                    question,
                    st.session_state.chat_history,
                )
                st.session_state.chat_history.append(
                    {"question": question, "answer": result["answer"], "sources": result["sources"]}
                )
                st.rerun()
            except Exception as e:
                st.error(t("answer_failed", e=e))


# ============================================================================
# Main
# ============================================================================

def main():
    inject_css()
    init_session_state()

    if get_lang() == "ar":
        inject_rtl_overrides()

    groq_api_key = get_groq_api_key()
    gemini_api_key = get_gemini_api_key()

    groq_client = get_groq_client(groq_api_key)
    gemini_client = get_gemini_client(gemini_api_key)
    embedding_model = load_embedding_model()

    render_sidebar()
    render_header()
    render_search_and_upload(groq_client, embedding_model)
    render_results(groq_client, embedding_model, gemini_client)
    render_summary()

    if st.session_state.vector_store is not None:
        reranker_model = load_reranker_model()
        render_chat(groq_client, reranker_model)


if __name__ == "__main__":
    main()

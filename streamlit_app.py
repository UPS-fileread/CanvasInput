import streamlit as st
import io

def extract_pdf_text(BytesIO) -> str:
    """Extract plain text from a PDF file-like object. Returns concatenated page texts.
    Uses PyPDF2 if available; shows a Streamlit error if not installed.
    """
    try:
        import PyPDF2  # type: ignore
    except Exception as e:
        st.error("PyPDF2 is required to parse PDFs. Install it with: pip install PyPDF2")
        raise
    try:
        BytesIO.seek(0)
        reader = PyPDF2.PdfReader(BytesIO)
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            # Add a simple page header separator to preserve boundaries
            pages.append(txt.strip())
        return "\n".join([p for p in pages if p])
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        raise

uploaded_files = st.file_uploader("...or upload one or more .txt / .jsonl / .pdf files", type=["txt", "jsonl", "pdf"], accept_multiple_files=True) 

if uploaded_files:
    combined_payloads = []
    for uploaded in uploaded_files:
        try:
            if uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf"):
                payload = extract_pdf_text(uploaded)
                if not payload.strip():
                    st.warning(f"{uploaded.name}: The PDF appears to contain no extractable text (it may be scanned). Try an OCR'd PDF.")
                combined_payloads.append(payload)
                try:
                    import PyPDF2  # type: ignore
                    uploaded.seek(0)
                    n_pages = len(PyPDF2.PdfReader(uploaded).pages)
                except Exception:
                    n_pages = None
                if n_pages:
                    st.success(f"Loaded {uploaded.name} — {n_pages} page(s)")
                else:
                    st.success(f"Loaded {uploaded.name}")
            else:
                payload = uploaded.read().decode("utf-8", errors="ignore")
                combined_payloads.append(payload)
                st.success(f"Loaded {uploaded.name} ({len(payload)} bytes)")
        except Exception as e:
            st.error(f"Failed to read {uploaded.name}: {e}")
    # Join with double newline to separate docs from different files
    st.session_state["corpus_input"] = "\n\n".join(combined_payloads)

st.text_area(
    "Paste documents (blank-line separated), JSONL (one {text} per line), or upload a PDF on the left",
    # ... rest of the parameters remain unchanged
)

import os
import json
import time
from typing import List, Dict, Any

import streamlit as st

# Import the builder from your package
try:
    from agentActions.ChronologyBuilder import ChronologyBuilder
except Exception as e:
    st.error("Failed to import ChronologyBuilder. Make sure your PYTHONPATH includes the project root and the module path is correct (agentActions/ChronologyBuilder.py).\n\nError: %s" % e)
    st.stop()

st.set_page_config(page_title="Chronology Builder Demo", layout="wide")

# -------------------------------
# Helpers
# -------------------------------

def parse_corpus_input(raw: str) -> List[str]:
    """Split text input into a corpus list. Supports JSONL (one JSON object per line with a `text` field)
    or plain text separated by blank lines."""
    raw = raw.strip()
    if not raw:
        return []

    # Try JSONL first
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if all((ln.startswith("{") and ln.endswith("}")) for ln in lines):
        docs = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                txt = obj.get("text")
                if isinstance(txt, str) and txt.strip():
                    docs.append(txt.strip())
            except Exception:
                # Not valid JSONL; fall back to plain text mode
                docs = []
                break
        if docs:
            return docs

    # Plain text: split by two or more newlines
    chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
    return chunks

def dedupe_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for e in events or []:
        key = (e.get("who"), e.get("what"), e.get("when"), e.get("where"))
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def show_events_table(events: List[Dict[str, Any]], title: str):
    import pandas as pd
    if not events:
        st.info(f"No events in {title}.")
        return
    # Keep expected columns and a source column if present
    cols = ["who", "what", "when", "where"] + (["source"] if any("source" in e for e in events) else [])
    data = [
        {k: e.get(k) for k in cols}
        for e in events
    ]
    df = pd.DataFrame(data)
    st.subheader(title)
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_cluster_view(view_name: str, clusters: List[Dict[str, Any]]):
    st.markdown(f"### Clusters by **{view_name}** ({len(clusters)})")
    if not clusters:
        st.info("No clusters found.")
        return
    # Summary table
    import pandas as pd
    rows = [{"topic": c.get("topic"), "events": len(c.get("events", []))} for c in clusters]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Per-cluster details
    for c in clusters:
        with st.expander(f"{c.get('topic')} — {len(c.get('events', []))} event(s)", expanded=False):
            show_events_table(c.get("events", []), title=f"Cluster: {c.get('topic')}")


# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.title("Chronology Builder")
    st.caption("LLM-only clustering — returns both views: by person and by event.")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set in the environment. Set it before running to enable extraction/clustering.")

    st.markdown("**Input options**")
    demo = st.toggle("Load example corpus", value=True)

    st.markdown("---")
    st.markdown("**Run options**")
    include_sources = st.checkbox("Include sources in timeline output", value=True)

# -------------------------------
# Main area: Input
# -------------------------------
if demo:
    default_text = """
Bill Gates announced a new vaccine initiative in Seattle on 2023-05-12.

Elon Musk unveiled the new Tesla Roadster at the California event on 2023-02-10.

Satya Nadella presented Microsoft's AI strategy during the annual conference in 2023-09-01.
    """.strip()
else:
    default_text = ""

st.text_area(
    "Paste documents (plain text separated by blank lines, or JSONL with a `text` field per line)",
    value=default_text,
    height=240,
    key="corpus_input",
)

run = st.button("🚀 Build Chronology", type="primary")

# -------------------------------
# Execute
# -------------------------------
if run:
    docs = parse_corpus_input(st.session_state.get("corpus_input", ""))
    if not docs:
        st.warning("Please provide at least one document.")
        st.stop()

    st.write(f"**Docs:** {len(docs)}")
    if uploaded is not None and (uploaded.type == "application/pdf" or uploaded.name.lower().endswith(".pdf")):
        st.caption("Each uploaded PDF is treated as a single document (pages are concatenated). To create multiple docs, add blank lines between sections in the text box.")

    try:
        t0 = time.time()
        builder = ChronologyBuilder(corpus=docs)  # default 4-step pipeline
        state = builder.run()
        dt = time.time() - t0
    except Exception as e:
        st.exception(e)
        st.stop()

    # ---------------------------
    # Metrics
    # ---------------------------
    num_docs = len(state.get("docs", docs)) if isinstance(state.get("docs"), list) else len(docs)
    num_events = len(state.get("events", []))
    cl = state.get("clusters", {"person": [], "event": []})
    num_person_clusters = len(cl.get("person", []) or [])
    num_event_clusters = len(cl.get("event", []) or [])
    num_timeline = len(state.get("timeline", []))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Docs", num_docs)
    c2.metric("Events", num_events)
    c3.metric("Person clusters", num_person_clusters)
    c4.metric("Event clusters", num_event_clusters)
    c5.metric("Latency (s)", f"{dt:.2f}")

    # ---------------------------
    # Tabs: person/event clusters, timeline, raw
    # ---------------------------
    tab_person, tab_event, tab_timeline, tab_raw = st.tabs([
        "By Person", "By Event", "Timeline", "Raw JSON"
    ])

    with tab_person:
        show_cluster_view("person", cl.get("person", []) or [])

    with tab_event:
        show_cluster_view("event", cl.get("event", []) or [])

    with tab_timeline:
        tl_events = dedupe_events(state.get("timeline", []))
        if not include_sources:
            for e in tl_events:
                e.pop("source", None)
        show_events_table(tl_events, title="Chronology Timeline")

    with tab_raw:
        st.json(state)

else:
    st.info("Configure input on the left and click **Build Chronology**.")
"""
TODO: Cluster topics based on Issue and Person entities.
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional, Union

import os
import re
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

load_dotenv()
try:
    from openai import OpenAI  # Requires openai>=1.0.0
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# =============================================================================
# ChronologyBuilder (GPT-enabled):
#   corpus -> mine_docs -> extract_events -> cluster_topics -> propose_timeline
#   Default agent is GPTAgent (structured outputs). You can still pass your own
#   agent implementing the same hooks if desired.
# =============================================================================

# ---- Action models -----------------------------------------------------------

class MineDocsAction(BaseModel):
    kind: Literal["mine_docs"]
    query: Optional[str] = Field(
        default=None, description="Optional retrieval query override sent to the agent.search hook."
    )
    max_docs: Optional[int] = Field(
        default=None, ge=1, description="If set, limit the number of docs mined."
    )


class ExtractEventsAction(BaseModel):
    kind: Literal["extract_events"]
    fields: List[Literal["who", "what", "when", "where"]] = Field(
        default_factory=lambda: ["who", "what", "when", "where"],
        description="Event fields to extract from text.",
    )
    dedupe: bool = Field(
        default=True, description="Whether to deduplicate events using only the selected fields."
    )


class ClusterTopicsAction(BaseModel):
    kind: Literal["cluster_topics"]
    algorithm: Literal["auto", "gpt-topics", "kmeans", "agglomerative", "lda"] = Field(
        default="auto", description="Clustering strategy; default uses GPT to assign concise topics."
    )
    k: Optional[int] = Field(default=None, ge=1, description="Optional fixed k for clustering (unused for GPT).")


class ProposeTimelineAction(BaseModel):
    kind: Literal["propose_timeline"]
    granularity: Literal["auto", "day", "week", "month"] = Field(
        default="auto", description="Preferred ordering granularity; advisory for agent implementations."
    )
    include_sources: bool = Field(
        default=True, description="Attach source doc indices to timeline entries."
    )


ChronologyAction = Annotated[
    Union[MineDocsAction, ExtractEventsAction, ClusterTopicsAction, ProposeTimelineAction],
    Field(discriminator="kind"),
]


# ---- Dispatcher/registry -----------------------------------------------------

# A handler function processes a specific pipeline action, taking the action model,
# current state, and optional agent, and returns an updated state dictionary.
Handler = Callable[[BaseModel, Dict[str, Any], Optional[Any]], Dict[str, Any]]

# Registry mapping action kinds to their handler functions.
# Used by ChronologyBuilder.run for dynamic dispatch of actions.
HANDLERS: Dict[str, Handler] = {}


def register(kind: str):
    """
    Decorator factory to register a function as the handler for a given action kind.

    Inserts the decorated function into the HANDLERS registry under the specified kind.
    Used by decorating action handler functions to associate them with their action type.
    """
    def deco(fn: Handler):
        HANDLERS[kind] = fn
        return fn

    return deco


# ---- Utilities ---------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",  # 2024-07-31
    "%m/%d/%Y",  # 07/31/2024
    "%m/%d/%y",  # 07/31/24
]
_DATE_REGEX = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")


def _extract_first_date_iso(text: str) -> Optional[str]:
    """Return first date token normalized to ISO YYYY-MM-DD, if any."""
    if not text:
        return None
    m = _DATE_REGEX.search(text)
    if not m:
        return None
    raw = m.group(1)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return raw


def _parse_date_for_sort(value: Any) -> tuple[int, str]:
    """Return a sorting key for dates.

    (0, ISO) for parsed dates -> sorts before (1, raw_string) -> before (2, empty)
    This keeps stable ordering while being robust to missing/unparseable dates.
    """
    if not value:
        return (2, "")
    if isinstance(value, datetime):
        return (0, value.date().isoformat())
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return (0, dt.date().isoformat())
        except Exception:
            pass
    return (1, s)


# ---- GPT Agent ---------------------------------------------------------------

class Event(BaseModel):
    who: Optional[str] = None
    what: Optional[str] = None
    when: Optional[str] = None  # ISO 8601 preferred (YYYY-MM-DD)
    where: Optional[str] = None


class TopicList(BaseModel):
    topics: List[str]


class GPTAgent:
    """Agent that uses OpenAI to extract events and assign concise topic labels."""

    def __init__(self, model: Optional[str] = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if OpenAI is None:
            raise RuntimeError("openai python client not available")
        self.client = OpenAI()

    # Retrieval hook (no-op for now)
    def search(self, query: str, docs: List[str]) -> List[str]:
        return docs

    # ---- Extraction ---------------------------------------------------------
    def _build_extraction_messages(self, text: str) -> List[Dict[str, str]]:
        system = (
            "You are an information extraction service. Read the document and extract a single "
            "canonical event with fields: who, what (1–2 sentence summary), when (YYYY-MM-DD if possible), "
            "and where (if present). Use null when a field is not present. Do not invent details."
        )
        MAX_CHARS = 12000
        snippet = text if len(text) <= MAX_CHARS else text[:MAX_CHARS]
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Document:\n\n{snippet}"},
        ]

    def extract_events(self, docs: List[str], fields: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        keep = set(fields)
        for i, d in enumerate(docs):
            text = d or ""
            try:
                completion = self.client.chat.completions.parse(
                    model=self.model,
                    messages=self._build_extraction_messages(text),
                    response_format=Event,
                )
                event_obj: Event = completion.choices[0].message.parsed
                event = event_obj.model_dump()
            except Exception:
                # Fallback: minimal heuristic
                event = {
                    "who": None,
                    "what": text.strip().replace("\n", " ")[:180] + ("…" if len(text) > 180 else ""),
                    "when": _extract_first_date_iso(text),
                    "where": None,
                }
            filtered = {k: v for k, v in event.items() if k in keep}
            filtered["source"] = i
            out.append(filtered)
        return out

    # ---- Clustering ---------------------------------------------------------
    def _build_topic_messages(self, summaries: List[str]) -> List[Dict[str, str]]:
        system = (
            "You assign concise topic labels to events. Given a list of event summaries, "
            "return a short 2–4 word topic label for each item. Be consistent across similar items. "
            "Only output JSON with a 'topics' array of strings, same length as the input."
        )
        numbered = "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(summaries))
        user = f"Event summaries:\n\n{numbered}\n\nRespond with:\n{{\n  \"topics\": [\"topic for 1\", \"topic for 2\", ...]\n}}"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def cluster(self, events: List[Dict[str, Any]], algorithm: str, k: Optional[int]):
        # Use GPT topic labels regardless of algorithm (fallback to single cluster on error)
        if not events:
            return []
        summaries = [str(e.get("what") or "") for e in events]
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=self._build_topic_messages(summaries),
                response_format=TopicList,
            )
            topics = completion.choices[0].message.parsed.topics
            if len(topics) != len(events):
                raise ValueError("Topic count mismatch")
        except Exception:
            # Single-cluster fallback
            return [{"topic": "all", "events": events}]

        # Group by topic label
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for e, t in zip(events, topics):
            label = (t or "misc").strip() or "misc"
            buckets.setdefault(label, []).append(e)
        clusters = [{"topic": topic, "events": evs} for topic, evs in buckets.items()]
        return clusters

    # ---- Timeline -----------------------------------------------------------
    def propose_timeline(
        self, clusters: List[Dict[str, Any]], granularity: str, include_sources: bool
    ) -> List[Dict[str, Any]]:
        events = [e for c in clusters for e in c.get("events", [])]
        events.sort(key=lambda e: _parse_date_for_sort(e.get("when")))
        if not include_sources:
            for e in events:
                e.pop("source", None)
        return events


# ---- Action handlers ---------------------------------------------------------

@register("mine_docs")
def _mine_docs(a: MineDocsAction, state: Dict[str, Any], agent: Optional[Any]) -> Dict[str, Any]:
    docs = list(state.get("corpus", []))
    if agent and hasattr(agent, "search"):
        query = a.query or ""
        docs = agent.search(query, docs)
    if a.max_docs:
        docs = docs[: a.max_docs]
    return {**state, "docs": docs}


@register("extract_events")
def _extract_events(
    a: ExtractEventsAction, state: Dict[str, Any], agent: Optional[Any]
) -> Dict[str, Any]:
    docs = state.get("docs", state.get("corpus", [])) or []

    if agent and hasattr(agent, "extract_events"):
        events = agent.extract_events(docs, a.fields)
    else:
        # Minimal placeholder: create one empty event per doc with only requested fields
        events = []
        for i, _ in enumerate(docs):
            e: Dict[str, Any] = {"source": i}
            for f in a.fields:
                if f == "when":
                    e[f] = _extract_first_date_iso(str(_))
                elif f == "what":
                    s = str(_).strip().replace("\n", " ")
                    e[f] = (s[:180] + "…") if len(s) > 180 else s
                else:
                    e[f] = None
            events.append(e)

    if a.dedupe:
        seen = set()
        uniq: List[Dict[str, Any]] = []
        for e in events:
            key = tuple((f, e.get(f)) for f in a.fields)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        events = uniq

    return {**state, "events": events}


@register("cluster_topics")
def _cluster_topics(
    a: ClusterTopicsAction, state: Dict[str, Any], agent: Optional[Any]
) -> Dict[str, Any]:
    events = state.get("events", [])
    if agent and hasattr(agent, "cluster"):
        clusters = agent.cluster(events, a.algorithm, a.k)
    else:
        clusters = [{"topic": "auto", "events": events}]
    return {**state, "clusters": clusters}


@register("propose_timeline")
def _propose_timeline(
    a: ProposeTimelineAction, state: Dict[str, Any], agent: Optional[Any]
) -> Dict[str, Any]:
    clusters = state.get("clusters", []) or []

    if agent and hasattr(agent, "propose_timeline"):
        timeline = agent.propose_timeline(clusters, a.granularity, a.include_sources)
        return {**state, "timeline": timeline}

    # Fallback: flatten cluster events and sort by parsed date from the 'when' field
    events = [e for c in clusters for e in c.get("events", [])]
    events.sort(key=lambda e: _parse_date_for_sort(e.get("when")))
    if not a.include_sources:
        for e in events:
            e.pop("source", None)
    return {**state, "timeline": events}


# ---- Builder -----------------------------------------------------------------

class ChronologyBuilder(BaseModel):
    """
    Build a chronology of events from a corpus using a GPT-backed agent by default.

    The pipeline executes, in order:
      1) mine_docs          -> state["docs"]
      2) extract_events     -> state["events"]
      3) cluster_topics     -> state["clusters"]
      4) propose_timeline   -> state["timeline"]

    Provide an *agent* implementing optional hooks:
      - search(query, docs)
      - extract_events(docs, fields)
      - cluster(events, algorithm, k)
      - propose_timeline(clusters, granularity, include_sources)

    If no agent is supplied, GPTAgent is used.
    """

    corpus: List[str] = Field(
        default_factory=list, description="Raw documents or doc IDs to process."
    )

    actions: List[ChronologyAction] = Field(
        default_factory=lambda: [
            MineDocsAction(kind="mine_docs"),
            ExtractEventsAction(kind="extract_events"),
            ClusterTopicsAction(kind="cluster_topics"),
            ProposeTimelineAction(kind="propose_timeline"),
        ],
        description="Execution plan for the chronology pipeline.",
    )

    version: str = "2.0"

    @model_validator(mode="after")
    def _validate_sequence(self):  # type: ignore[override]
        order = {
            "mine_docs": 0,
            "extract_events": 1,
            "cluster_topics": 2,
            "propose_timeline": 3,
        }
        last = -1
        for a in self.actions:
            step = order[a.kind]
            if step < last:
                raise ValueError(
                    "Actions must follow: mine_docs → extract_events → cluster_topics → propose_timeline"
                )
            last = step
        return self

    def run(self, agent: Optional[Any] = None) -> Dict[str, Any]:
        """Execute the configured pipeline and return the final state (includes 'timeline')."""
        # Default to GPTAgent if none provided
        if agent is None:
            agent = GPTAgent()
        state: Dict[str, Any] = {"corpus": self.corpus}
        for action in self.actions:
            handler = HANDLERS[action.kind]
            state = handler(action, state, agent)
        return state


__all__ = [
    "MineDocsAction",
    "ExtractEventsAction",
    "ClusterTopicsAction",
    "ProposeTimelineAction",
    "ChronologyAction",
    "ChronologyBuilder",
    "GPTAgent",
]
"""
LLM-only clustering: extract {person, event} for each event, and return both views.
state["clusters"] = {"person": [...], "event": [...]}.
Multi-assign: if a summary mentions multiple people, its event is added to each person's cluster.
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


# ---- LLM Clustering output models ----
class TopicAssignment(BaseModel):
    # Back-compat: either a single person or a list of persons can be provided by the LLM
    person: Optional[str] = None
    persons: Optional[List[str]] = None
    event: Optional[str] = None

class TopicAssignments(BaseModel):
    clusters: List[TopicAssignment]


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
            "You assign cluster labels by identifying PEOPLE and an EVENT for each item.\n"
            "For every input summary, extract:\n"
            "  - persons: array of distinct person names mentioned ([] if none)\n"
            "  - event: a concise 2–4 word event label (string) or null if none is clear\n"
            "Return ONLY JSON with a 'clusters' array of objects matching the input length."
        )
        numbered = "\n".join(f"{idx+1}. {s}" for idx, s in enumerate(summaries))
        user = (
            "Event summaries:\n\n"
            f"{numbered}\n\n"
            "Respond with JSON exactly like:\n"
            "{\n"
            "  \"clusters\": [\n"
            "    {\"persons\": [\"Alice\", \"Bob\"], \"event\": \"product launch\"},\n"
            "    {\"persons\": [], \"event\": null}\n"
            "  ]\n"
            "}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def cluster(self, events: List[Dict[str, Any]]):
        """LLM-only clustering. Returns both views: clusters by person and clusters by event.
        If multiple people are mentioned in a summary, the event is added to *each* person's cluster.
        Output shape:
        {
          "person": [{"topic": <person or \"anyone\">, "person": <str|None>, "event": None, "events": [...]}, ...],
          "event":  [{"topic": <event or \"general\">, "person": None, "event": <str|None>, "events": [...]}, ...]
        }
        """
        if not events:
            return {"person": [], "event": []}

        summaries = [str(e.get("what") or "") for e in events]
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=self._build_topic_messages(summaries),
                response_format=TopicAssignments,
            )
            assigns: List[TopicAssignment] = completion.choices[0].message.parsed.clusters
            if len(assigns) != len(events):
                raise ValueError("Assignment count mismatch")
        except Exception:
            return {
                "person": [{"topic": "anyone", "person": None, "event": None, "events": events}],
                "event":  [{"topic": "general", "person": None, "event": None, "events": events}],
            }

        def _norm_str(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = re.sub(r"\s+", " ", str(x)).strip().lower()
            return s or None

        def _norm_list(xs: Optional[List[str]]) -> List[str]:
            out: List[str] = []
            if not xs:
                return out
            for x in xs:
                nx = _norm_str(x)
                if nx and nx not in out:
                    out.append(nx)
            return out

        by_person: Dict[Optional[str], List[Dict[str, Any]]] = {}
        by_event: Dict[Optional[str], List[Dict[str, Any]]] = {}

        for e, a in zip(events, assigns):
            # Prefer array if present, else back-compat single string
            persons = _norm_list(a.persons) or ([] if a.person is None else [_norm_str(a.person)])
            event_label = _norm_str(a.event)

            # Fan-out: add the event under every mentioned person; if none, bucket under None (anyone)
            if persons:
                for p in persons:
                    by_person.setdefault(p, []).append(e)
            else:
                by_person.setdefault(None, []).append(e)

            # Event view: standard single label bucketing (None -> "general")
            by_event.setdefault(event_label, []).append(e)

        person_clusters: List[Dict[str, Any]] = []
        for p, evs in by_person.items():
            person_clusters.append({
                "topic": p or "anyone",
                "person": p,
                "event": None,
                "events": evs,
            })

        event_clusters: List[Dict[str, Any]] = []
        for ev, evs in by_event.items():
            event_clusters.append({
                "topic": ev or "general",
                "person": None,
                "event": ev,
                "events": evs,
            })

        return {"person": person_clusters, "event": event_clusters}

    # ---- Timeline -----------------------------------------------------------
    def propose_timeline(
        self, clusters: List[Dict[str, Any]] | Dict[str, List[Dict[str, Any]]], granularity: str, include_sources: bool
    ) -> List[Dict[str, Any]]:
        # Accept list of clusters OR dict with keys {"person": [...], "event": [...]}.
        if isinstance(clusters, dict):
            all_clusters: List[Dict[str, Any]] = []
            for v in clusters.values():
                all_clusters.extend(v or [])
        else:
            all_clusters = clusters or []
        events = [e for c in all_clusters for e in c.get("events", [])]
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
        clusters = agent.cluster(events)  # returns {"person": [...], "event": [...]}
    else:
        clusters = {"person": [{"topic": "anyone", "person": None, "event": None, "events": events}],
                    "event":  [{"topic": "general", "person": None, "event": None, "events": events}]}
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
      - cluster(events)                  # returns {"person": [...], "event": [...]}
      - propose_timeline(clusters, granularity, include_sources)

    state["clusters"] is a dict with keys "person" and "event", each a list of clusters.

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
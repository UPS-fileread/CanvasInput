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
    algorithm: Literal["auto", "gpt-topics", "event-incremental"] = Field(
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

# For incremental event tagging decisions
class TagDecision(BaseModel):
    # Exactly one of these should be non-null
    use_tag: Optional[str] = None   # Must match an existing tag exactly (case-insensitive)
    new_tag: Optional[str] = None   # Provide a concise 2–4 word tag when no existing tag fits


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

    def _build_incremental_messages(self, summary: str, existing_tags: List[str]) -> List[Dict[str, str]]:
        system = (
            "You are an event tagger. Given an event summary and a list of existing event tags, either: "
            "(a) pick ONE of the existing tags that best fits, returning it exactly, or (b) propose a NEW concise 2–4 word tag. "
            "Return ONLY JSON matching {\"use_tag\": string|null, \"new_tag\": string|null}. "
            "If you choose an existing tag, it must be returned EXACTLY as provided (case-insensitive match acceptable). "
            "Prefer reuse over creating new tags unless none fit. Make tags short and general, avoid punctuation."
        )
        tag_list = "\n".join(f"- {t}" for t in existing_tags) if existing_tags else "(none)"
        user = (
            "Existing tags:\n" + tag_list + "\n\n" +
            "Event summary:\n" + summary + "\n\n" +
            "Respond with JSON only."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def cluster_events(self, events: List[Dict[str, Any]], algorithm: str = "event-incremental") -> List[Dict[str, Any]]:
        """Return event-centric clusters only.
        Default uses incremental tagging ("event-incremental"). If algorithm == "gpt-topics",
        it falls back to the batch LLM path and extracts event labels only.
        """
        if not events:
            return []

        def _norm_str(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = re.sub(r"\s+", " ", str(x)).strip()
            return s if s else None

        if algorithm == "gpt-topics":
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
                return [{"topic": "general", "person": None, "event": None, "events": events}]

            by_event: Dict[Optional[str], List[Dict[str, Any]]] = {}
            for e, a in zip(events, assigns):
                ev = _norm_str(a.event)
                by_event.setdefault(ev, []).append(e)

            out: List[Dict[str, Any]] = []
            for ev, evs in by_event.items():
                out.append({"topic": ev or "general", "person": None, "event": ev, "events": evs})
            return out

        # Default incremental path (event-incremental)
        existing_tags: List[str] = []
        by_event: Dict[str, List[Dict[str, Any]]] = {}
        for e in events:
            summary = str(e.get("what") or "")
            try:
                completion = self.client.chat.completions.parse(
                    model=self.model,
                    messages=self._build_incremental_messages(summary, existing_tags),
                    response_format=TagDecision,
                )
                decision: TagDecision = completion.choices[0].message.parsed
                use_tag = _norm_str(decision.use_tag)
                new_tag = _norm_str(decision.new_tag)
            except Exception:
                words = [w for w in re.split(r"\W+", summary.lower()) if w]
                new_tag = " ".join(words[:3]) or "general"
                use_tag = None

            if use_tag:
                match = None
                for t in existing_tags:
                    if t.lower() == use_tag.lower():
                        match = t
                        break
                tag = match or use_tag
                if tag not in existing_tags:
                    existing_tags.append(tag)
            else:
                tag = new_tag or "general"
                if tag not in existing_tags:
                    existing_tags.append(tag)

            by_event.setdefault(tag, []).append(e)

        event_clusters: List[Dict[str, Any]] = []
        for ev, evs in by_event.items():
            event_clusters.append({"topic": ev, "person": None, "event": ev, "events": evs})
        return event_clusters

    def cluster_persons(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return person-centric clusters using the batch gpt-topics path only."""
        if not events:
            return []

        def _norm_str(x: Optional[str]) -> Optional[str]:
            if x is None:
                return None
            s = re.sub(r"\s+", " ", str(x)).strip()
            return s if s else None

        def _norm_list(xs: Optional[List[str]]) -> List[str]:
            out: List[str] = []
            if not xs:
                return out
            for x in xs:
                nx = _norm_str(x)
                if nx and nx.lower() not in [o.lower() for o in out]:
                    out.append(nx)
            return out

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
            return [{"topic": "anyone", "person": None, "event": None, "events": events}]

        by_person: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for e, a in zip(events, assigns):
            persons = _norm_list(a.persons) or ([] if a.person is None else [_norm_str(a.person)])
            if persons:
                for p in persons:
                    by_person.setdefault(p, []).append(e)
            else:
                by_person.setdefault(None, []).append(e)

        out: List[Dict[str, Any]] = []
        for p, evs in by_person.items():
            out.append({"topic": p or "anyone", "person": p, "event": None, "events": evs})
        return out

    def cluster(self, events: List[Dict[str, Any]], algorithm: str = "auto"):
        if not events:
            return {"person": [], "event": []}

        # Orchestration rules (supported algorithms):
        # - "gpt-topics": one LLM pass to get both views
        # - "event-incremental": incremental tags for event view only
        # - "auto" (default): person view via gpt-topics; event view via incremental
        if algorithm == "gpt-topics":
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
                s = re.sub(r"\s+", " ", str(x)).strip()
                return s if s else None

            def _norm_list(xs: Optional[List[str]]) -> List[str]:
                out: List[str] = []
                if not xs:
                    return out
                for x in xs:
                    nx = _norm_str(x)
                    if nx and nx.lower() not in [o.lower() for o in out]:
                        out.append(nx)
                return out

            by_person: Dict[Optional[str], List[Dict[str, Any]]] = {}
            by_event: Dict[Optional[str], List[Dict[str, Any]]] = {}
            for e, a in zip(events, assigns):
                persons = _norm_list(a.persons) or ([] if a.person is None else [_norm_str(a.person)])
                ev = _norm_str(a.event)
                if persons:
                    for p in persons:
                        by_person.setdefault(p, []).append(e)
                else:
                    by_person.setdefault(None, []).append(e)
                by_event.setdefault(ev, []).append(e)

            person_clusters: List[Dict[str, Any]] = []
            for p, evs in by_person.items():
                person_clusters.append({"topic": p or "anyone", "person": p, "event": None, "events": evs})
            event_clusters: List[Dict[str, Any]] = []
            for ev, evs in by_event.items():
                event_clusters.append({"topic": ev or "general", "person": None, "event": ev, "events": evs})
            return {"person": person_clusters, "event": event_clusters}

        if algorithm == "event-incremental":
            return {"person": [], "event": self.cluster_events(events, "event-incremental")}

        # auto
        return {
            "person": self.cluster_persons(events),
            "event": self.cluster_events(events, "event-incremental"),
        }

    # ---- Timeline -----------------------------------------------------------
    def propose_timeline(
        self, clusters: List[Dict[str, Any]] | Dict[str, List[Dict[str, Any]]], granularity: str, include_sources: bool
    ) -> List[Dict[str, Any]]:
        # Accept list of clusters OR dict with keys {"person": [...], "event": [...]}. De-duplicate events across views.
        if isinstance(clusters, dict):
            all_clusters: List[Dict[str, Any]] = []
            for v in clusters.values():
                all_clusters.extend(v or [])
        else:
            all_clusters = clusters or []

        # Flatten with de-duplication
        events: List[Dict[str, Any]] = []
        seen: set = set()

        def _event_key(e: Dict[str, Any]):
            # Prefer unique source index if present; else fall back to a tuple of identifying fields
            src = e.get("source")
            if src is not None:
                return ("source", src)
            return (
                "fields",
                str(e.get("who") or "").strip(),
                str(e.get("what") or "").strip(),
                str(e.get("when") or "").strip(),
                str(e.get("where") or "").strip(),
            )

        for c in all_clusters:
            for e in c.get("events", []):
                k = _event_key(e)
                if k in seen:
                    continue
                seen.add(k)
                events.append(e)

        # Sort by parsed date
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
        clusters = agent.cluster(events, a.algorithm)  # returns {"person": [...], "event": []}
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

    # Fallback: flatten cluster events (dict or list) with de-dup and sort by date
    if isinstance(clusters, dict):
        all_clusters = []
        for v in clusters.values():
            all_clusters.extend(v or [])
    else:
        all_clusters = clusters or []

    events: List[Dict[str, Any]] = []
    seen = set()

    def _event_key(e: Dict[str, Any]):
        src = e.get("source")
        if src is not None:
            return ("source", src)
        return (
            "fields",
            str(e.get("who") or "").strip(),
            str(e.get("what") or "").strip(),
            str(e.get("when") or "").strip(),
            str(e.get("where") or "").strip(),
        )

    for c in all_clusters:
        for e in c.get("events", []):
            k = _event_key(e)
            if k in seen:
                continue
            seen.add(k)
            events.append(e)

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
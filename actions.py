"""
TODO: Move to Langfuse.

based on the actions each classes has, decide which class to use.
"""
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Union, Literal, Callable, Dict, Any

# --- Chronology actions & dispatch registry ---

class MineDocsAction(BaseModel):
    kind: Literal["mine_docs"]
    query: str | None = Field(default=None, description="Optional retrieval query override.")
    max_docs: int | None = Field(default=None, ge=1, description="Limit number of docs mined.")

class ExtractEventsAction(BaseModel):
    kind: Literal["extract_events"]
    fields: list[Literal["who", "what", "when", "where"]] = Field(
        default_factory=lambda: ["who", "what", "when", "where"],
        description="Event fields to extract from text.",
    )
    dedupe: bool = Field(default=True, description="Whether to deduplicate near-identical events.")

class ClusterTopicsAction(BaseModel):
    kind: Literal["cluster_topics"]
    algorithm: Literal["auto", "kmeans", "agglomerative", "lda"] = Field(
        default="auto", description="Clustering strategy."
    )
    k: int | None = Field(default=None, ge=1, description="Optional fixed number of clusters.")

class ProposeTimelineAction(BaseModel):
    kind: Literal["propose_timeline"]
    granularity: Literal["auto", "day", "week", "month"] = Field(default="auto")
    include_sources: bool = Field(default=True, description="Attach source doc refs to timeline entries.")

ChronologyAction = Annotated[
    Union[MineDocsAction, ExtractEventsAction, ClusterTopicsAction, ProposeTimelineAction],
    Field(discriminator="kind"),
]

# Simple dispatcher (handlers are placeholders; wire to your real agent/tools)
Handler = Callable[[BaseModel, dict[str, Any], Any | None], dict[str, Any]]
HANDLERS: Dict[str, Handler] = {}

def register(kind: str):
    def deco(fn: Handler):
        HANDLERS[kind] = fn
        return fn
    return deco

@register("mine_docs")
def _mine_docs(a: MineDocsAction, state: dict[str, Any], agent: Any | None) -> dict[str, Any]:
    # Replace with retrieval against your doc store / vector index
    docs = state.get("corpus", [])
    if agent and hasattr(agent, "search"):
        docs = agent.search(a.query, docs)  # user-defined hook
    if a.max_docs:
        docs = docs[: a.max_docs]
    return {**state, "docs": docs}

@register("extract_events")
def _extract_events(a: ExtractEventsAction, state: dict[str, Any], agent: Any | None) -> dict[str, Any]:
    docs = state.get("docs", state.get("corpus", []))
    # Replace with your IE pipeline or LLM tool call
    events = []
    if agent and hasattr(agent, "extract_events"):
        events = agent.extract_events(docs, a.fields)
    else:
        # placeholder structure
        events = [{"who": "", "what": "", "when": "", "where": "", "source": i} for i, _ in enumerate(docs)]
    if a.dedupe:
        # naive placeholder dedupe
        seen = set()
        uniq = []
        for e in events:
            key = (e.get("who"), e.get("what"), e.get("when"), e.get("where"))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(e)
        events = uniq
    return {**state, "events": events}

@register("cluster_topics")
def _cluster_topics(a: ClusterTopicsAction, state: dict[str, Any], agent: Any | None) -> dict[str, Any]:
    events = state.get("events", [])
    clusters = []
    if agent and hasattr(agent, "cluster"):
        clusters = agent.cluster(events, a.algorithm, a.k)
    else:
        # placeholder single-cluster
        clusters = [{"topic": "auto", "events": events}]
    return {**state, "clusters": clusters}

@register("propose_timeline")
def _propose_timeline(a: ProposeTimelineAction, state: dict[str, Any], agent: Any | None) -> dict[str, Any]:
    clusters = state.get("clusters", [])
    timeline = []
    if agent and hasattr(agent, "propose_timeline"):
        timeline = agent.propose_timeline(clusters, a.granularity, a.include_sources)
    else:
        # placeholder: flatten cluster events and sort by 'when' if present
        events = [e for c in clusters for e in c.get("events", [])]
        timeline = sorted(events, key=lambda e: str(e.get("when", "")))
    return {**state, "timeline": timeline}

class ChronologyBuilder(BaseModel):
    """
    Pydantic model for building a chronlogy of events.

    Agent: mines docs → extracts events (who/what/when/where) → clusters by topic → proposes a draft timeline.
    """

    # Inputs to the pipeline (seed corpus or doc IDs)
    corpus: list[str] = Field(default_factory=list, description="Raw docs or doc IDs to process.")

    # Default action plan (can be overridden)
    actions: list[ChronologyAction] = Field(
        default_factory=lambda: [
            MineDocsAction(kind="mine_docs"),
            ExtractEventsAction(kind="extract_events"),
            ClusterTopicsAction(kind="cluster_topics"),
            ProposeTimelineAction(kind="propose_timeline"),
        ],
        description="Execution plan for the chronology pipeline.",
    )

    version: str = "1.0"

    @model_validator(mode="after")
    def _validate_sequence(self):
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

    def run(self, agent: Any | None = None) -> dict[str, Any]:
        """
        Execute the pipeline with the provided agent/tooling. Returns final state containing 'timeline'.
        """
        state: dict[str, Any] = {"corpus": self.corpus}
        for action in self.actions:
            handler = HANDLERS[action.kind]
            state = handler(action, state, agent)
        return state

class DepositionPrepPack(BaseModel):
    """
    Pydantic model for building a chronlogy of events.
    """
    pass

class MotionSkeleton(BaseModel):
    """
    Pydantic model for building a chronlogy of events.
    """
    pass

class IssueFocusedHotDocTriage(BaseModel):
    """
    Pydantic model for building a chronlogy of events.
    """
    pass

class PrivilegeRiskSweep(BaseModel):
    """
    Pydantic model for building a chronlogy of events.
    """
    pass

class ExhibitBinderAutoComposer(BaseModel):
    """
    Pydantic model for building a chronlogy of events.
    """
    pass

class FactConsistencyChecker(BaseModel):
    """
    
    """

    pass

class CrossMatterReuseFinder(BaseModel):
    """
    Pydantic model
    """
    
    pass

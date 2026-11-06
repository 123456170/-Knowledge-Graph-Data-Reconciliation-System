from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
import time
import tracemalloc
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import networkx as nx
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# -----------------------------------------------------------------------------
# Constants, error codes, and strict base model
# -----------------------------------------------------------------------------

APP_VERSION = "018.1.1"
SCHEMA_VERSION = "kgdr-018.1"
DEFAULT_SEED = 20261106
ARTIFACT_DIR = Path("artifacts")

SRC_A = "SRC_A"
SRC_B = "SRC_B"

ERROR_CODES = {
    "schema_version": "KGDR-ERR-001",
    "unknown_field": "KGDR-ERR-002",
    "invalid_value": "KGDR-ERR-003",
    "empty_input": "KGDR-ERR-004",
    "duplicate_id": "KGDR-ERR-005",
    "unsafe_path": "KGDR-ERR-006",
    "artifact_write": "KGDR-ERR-007",
    "validation_failed": "KGDR-ERR-008",
    "unexpected": "KGDR-ERR-999",
}


class AppError(Exception):
    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# -----------------------------------------------------------------------------
# Domain models
# -----------------------------------------------------------------------------


class Provenance(StrictModel):
    schema_version: str = SCHEMA_VERSION
    source_document: str
    span_start: int = Field(0, ge=0)
    span_end: int = Field(0, ge=0)
    extractor_version: str = "synthetic-rule-1.0"
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    ingested_at: datetime

    @field_validator("ingested_at", mode="after")
    @classmethod
    def ensure_tz_ingested(cls, v: datetime) -> datetime:
        return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)

    @model_validator(mode="after")
    def validate_span(self) -> "Provenance":
        if self.span_end < self.span_start:
            raise ValueError("span_end must be >= span_start")
        if not self.source_document.strip():
            raise ValueError("source_document must not be empty")
        return self


class PropertyValue(StrictModel):
    schema_version: str = SCHEMA_VERSION
    value: str
    observed_at: datetime
    confidence: float = Field(0.9, ge=0.0, le=1.0)
    provenance: Provenance

    @field_validator("observed_at", mode="after")
    @classmethod
    def ensure_tz_observed(cls, v: datetime) -> datetime:
        return v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)

    @field_validator("value")
    @classmethod
    def value_not_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("value must not be empty")
        return str(v).strip()


class Node(StrictModel):
    schema_version: str = SCHEMA_VERSION
    node_id: str
    source_id: str
    label: str
    category: str
    aliases: List[str] = Field(default_factory=list)
    properties: Dict[str, List[PropertyValue]] = Field(default_factory=dict)
    provenance: Provenance

    @field_validator("node_id", "source_id", "label", "category")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("field must not be empty")
        return str(v).strip()


class Edge(StrictModel):
    schema_version: str = SCHEMA_VERSION
    edge_id: str
    source_id: str
    source_node_id: str
    target_node_id: str
    relation: str
    properties: Dict[str, str] = Field(default_factory=dict)
    provenance: Provenance

    @field_validator("edge_id", "source_id", "source_node_id", "target_node_id", "relation")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("field must not be empty")
        return str(v).strip()


class GraphInput(StrictModel):
    schema_version: str = SCHEMA_VERSION
    source_id: str
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    @field_validator("source_id")
    @classmethod
    def source_non_empty(cls, v: str) -> str:
        if not str(v).strip():
            raise ValueError("source_id must not be empty")
        return str(v).strip()

    @model_validator(mode="after")
    def validate_graph(self) -> "GraphInput":
        seen: set[str] = set()

        for n in self.nodes:
            if n.source_id != self.source_id:
                raise ValueError(f"node {n.node_id} source_id does not match graph source_id")
            if n.node_id in seen:
                raise ValueError(f"duplicate node_id: {n.node_id}")
            seen.add(n.node_id)

        for e in self.edges:
            if e.source_id != self.source_id:
                raise ValueError(f"edge {e.edge_id} source_id does not match graph source_id")
            if e.source_node_id not in seen:
                raise ValueError(f"edge {e.edge_id} references missing source node {e.source_node_id}")
            if e.target_node_id not in seen:
                raise ValueError(f"edge {e.edge_id} references missing target node {e.target_node_id}")

        return self


class MatchCandidate(StrictModel):
    schema_version: str = SCHEMA_VERSION
    left_source: str
    left_id: str
    right_source: str
    right_id: str
    left_label: str
    right_label: str
    score: float = Field(ge=0.0, le=1.0)
    id_score: float = Field(ge=0.0, le=1.0)
    label_score: float = Field(ge=0.0, le=1.0)
    alias_score: float = Field(ge=0.0, le=1.0)
    embedding_score: float = Field(ge=0.0, le=1.0)
    neighborhood_score: float = Field(ge=0.0, le=1.0)
    band: Literal["auto_accept", "review", "reject"]
    evidence: List[str] = Field(default_factory=list)


class Claim(StrictModel):
    schema_version: str = SCHEMA_VERSION
    source_id: str
    node_id: str
    property_name: str
    value: str
    observed_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    reliability: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    provenance: Provenance


class ConflictRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    pair_id: str
    left_source: str
    left_id: str
    right_source: str
    right_id: str
    property_name: str
    conflict_type: Literal["error", "temporal", "perspective"]
    functional: bool
    claims: List[Claim]
    weighted_winner: Optional[str] = None
    majority_winner: Optional[str] = None
    agreement: bool = False
    status: Literal["resolved", "review"] = "review"
    rationale: str = ""


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def log_event(logs: List[Dict[str, Any]], level: str, event: str, **details: Any) -> None:
    logs.append(
        {
            "timestamp": utcnow().isoformat(),
            "level": level,
            "event": event,
            "details": details,
        }
    )


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint_graph(graph: GraphInput) -> str:
    return sha256_text(canonical_json(graph.model_dump(mode="json")))


def config_hash(cfg: Dict[str, Any]) -> str:
    return sha256_text(canonical_json(cfg))


def uid(source_id: str, node_id: str) -> str:
    return f"{source_id}:{node_id}"


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    text = strip_accents(str(text)).lower()
    text = re.sub(r"[^a-z0-9\s'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_value(property_name: str, value: str) -> str:
    if property_name == "phone":
        return re.sub(r"\D", "", str(value))
    return normalize_text(value)


def validate_property_value(property_name: str, value: str) -> bool:
    v = str(value).strip()
    if not v:
        return False
    if property_name == "email":
        return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v))
    if property_name == "phone":
        digits = re.sub(r"\D", "", v)
        return 7 <= len(digits) <= 15
    return True


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# -----------------------------------------------------------------------------
# Deterministic local similarity helpers
# -----------------------------------------------------------------------------


def char_ngrams(text: str, n: int = 3) -> List[str]:
    text = f" {normalize_text(text)} "
    if len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def pseudo_embedding(text: str, dim: int = 64) -> List[float]:
    """
    Deterministic local surrogate embedding.

    This is not a neural embedding model. It is a reproducible hashed
    character/word n-gram vector used only for offline CPU demo behaviour.
    """
    vec = [0.0] * dim
    tokens = char_ngrams(text) + normalize_text(text).split()

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, byteorder="big")
        idx = h % dim
        sign = 1.0 if ((h >> 63) & 1) else -1.0
        vec[idx] += sign

    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------


def build_nx_graph(graph: GraphInput) -> nx.Graph:
    g = nx.Graph()

    for n in graph.nodes:
        g.add_node(
            uid(graph.source_id, n.node_id),
            label=n.label,
            source=graph.source_id,
            category=n.category,
        )

    for e in graph.edges:
        g.add_edge(
            uid(e.source_id, e.source_node_id),
            uid(e.source_id, e.target_node_id),
            relation=e.relation,
        )

    return g


def neighbor_signature(g: nx.Graph, node_uid: str) -> set[str]:
    sig: set[str] = set()
    if node_uid not in g:
        return sig

    for neighbor in g.neighbors(node_uid):
        rel = g[node_uid][neighbor].get("relation", "related")
        lbl = g.nodes[neighbor].get("label", "")
        sig.add(f"{normalize_text(rel)}:{normalize_text(lbl)}")

    return sig


def latest_claim(node: Node, property_name: str) -> Optional[PropertyValue]:
    claims = node.properties.get(property_name, [])
    if not claims:
        return None
    return max(claims, key=lambda c: c.observed_at)


def all_unique_claims(node: Node, property_name: str) -> List[PropertyValue]:
    out: Dict[str, PropertyValue] = {}
    for c in node.properties.get(property_name, []):
        key = normalize_value(property_name, c.value)
        if key not in out or c.observed_at > out[key].observed_at:
            out[key] = c
    return list(out.values())


def node_text(node: Node) -> str:
    parts: List[str] = [node.label, node.category]
    parts.extend(node.aliases)

    for prop in ["employee_id", "legal_id", "email", "role", "legal_name", "industry", "location"]:
        c = latest_claim(node, prop)
        if c is not None:
            parts.append(f"{prop}={c.value}")

    return " ".join(parts)


# -----------------------------------------------------------------------------
# Synthetic demo data
# -----------------------------------------------------------------------------


def _prov(doc: str, conf: float = 0.9, start: int = 0, end: int = 120) -> Provenance:
    return Provenance(
        source_document=doc,
        span_start=start,
        span_end=end,
        extractor_version="synthetic-rule-1.0",
        confidence=conf,
        ingested_at=_dt("2026-01-15T09:00:00+00:00"),
    )


def _pv(value: str, observed: str, conf: float, doc: str) -> PropertyValue:
    return PropertyValue(
        value=value,
        observed_at=_dt(observed),
        confidence=conf,
        provenance=_prov(doc, conf),
    )


def make_demo_graphs() -> Tuple[GraphInput, GraphInput]:
    nodes_a = [
        Node(
            node_id="A1",
            source_id=SRC_A,
            label="Alice Johnson",
            category="Person",
            aliases=["Alice Jonson", "A. Johnson"],
            properties={
                "employee_id": [_pv("EMP-1001", "2024-01-10T00:00:00+00:00", 0.99, "hr_export_2024.csv")],
                "email": [_pv("alice.johnson@acme.example", "2024-06-01T00:00:00+00:00", 0.93, "crm_export.csv")],
                "role": [_pv("Software Engineer", "2024-06-01T00:00:00+00:00", 0.90, "hr_export_2024.csv")],
                "location": [_pv("New York", "2024-06-01T00:00:00+00:00", 0.85, "hr_export_2024.csv")],
                "phone": [_pv("+1-555-0101", "2024-06-01T00:00:00+00:00", 0.80, "crm_export.csv")],
            },
            provenance=_prov("hr_export_2024.csv", 0.95),
        ),
        Node(
            node_id="A2",
            source_id=SRC_A,
            label="Bob Smith",
            category="Person",
            aliases=["Robert Smith"],
            properties={
                "employee_id": [_pv("EMP-1002", "2024-01-10T00:00:00+00:00", 0.99, "hr_export_2024.csv")],
                "email": [_pv("bob.smith@acme.example", "2024-06-01T00:00:00+00:00", 0.93, "crm_export.csv")],
                "role": [_pv("Data Analyst", "2024-06-01T00:00:00+00:00", 0.91, "hr_export_2024.csv")],
                "location": [_pv("Boston", "2024-06-01T00:00:00+00:00", 0.88, "hr_export_2024.csv")],
            },
            provenance=_prov("hr_export_2024.csv", 0.95),
        ),
        Node(
            node_id="A3",
            source_id=SRC_A,
            label="Carol Diaz",
            category="Person",
            aliases=["Carol D'Az"],
            properties={
                "employee_id": [_pv("EMP-1003", "2024-01-10T00:00:00+00:00", 0.99, "hr_export_2024.csv")],
                "email": [_pv("carol.diaz@acme.example", "2024-06-01T00:00:00+00:00", 0.94, "crm_export.csv")],
                "role": [_pv("Engineering Manager", "2024-06-01T00:00:00+00:00", 0.95, "hr_export_2024.csv")],
                "location": [_pv("Austin", "2024-06-01T00:00:00+00:00", 0.89, "hr_export_2024.csv")],
            },
            provenance=_prov("hr_export_2024.csv", 0.95),
        ),
        Node(
            node_id="A4",
            source_id=SRC_A,
            label="Acme Corp",
            category="Organization",
            aliases=["ACME", "Acme Corporation"],
            properties={
                "legal_id": [_pv("ORG-ACME", "2024-01-10T00:00:00+00:00", 0.99, "org_master.csv")],
                "legal_name": [_pv("Acme Corp", "2024-01-10T00:00:00+00:00", 0.95, "org_master.csv")],
                "industry": [
                    _pv("Software", "2024-01-10T00:00:00+00:00", 0.92, "org_master.csv"),
                    _pv("Manufacturing", "2024-01-10T00:00:00+00:00", 0.70, "legacy_org.csv"),
                ],
                "location": [_pv("New York", "2024-01-10T00:00:00+00:00", 0.90, "org_master.csv")],
            },
            provenance=_prov("org_master.csv", 0.96),
        ),
        Node(
            node_id="A5",
            source_id=SRC_A,
            label="Globex Corporation",
            category="Organization",
            aliases=["Globex"],
            properties={
                "legal_id": [_pv("ORG-GLOBEX", "2024-01-10T00:00:00+00:00", 0.98, "org_master.csv")],
                "legal_name": [_pv("Globex Corporation", "2024-01-10T00:00:00+00:00", 0.95, "org_master.csv")],
                "industry": [_pv("Manufacturing", "2024-01-10T00:00:00+00:00", 0.91, "org_master.csv")],
            },
            provenance=_prov("org_master.csv", 0.96),
        ),
    ]

    edges_a = [
        Edge(
            edge_id="EA1",
            source_id=SRC_A,
            source_node_id="A1",
            target_node_id="A4",
            relation="works_at",
            provenance=_prov("hr_export_2024.csv", 0.97),
        ),
        Edge(
            edge_id="EA2",
            source_id=SRC_A,
            source_node_id="A2",
            target_node_id="A4",
            relation="works_at",
            provenance=_prov("hr_export_2024.csv", 0.97),
        ),
        Edge(
            edge_id="EA3",
            source_id=SRC_A,
            source_node_id="A3",
            target_node_id="A1",
            relation="manages",
            provenance=_prov("org_chart.json", 0.94),
        ),
        Edge(
            edge_id="EA4",
            source_id=SRC_A,
            source_node_id="A4",
            target_node_id="A5",
            relation="competitor",
            provenance=_prov("market_report.pdf", 0.82),
        ),
    ]

    nodes_b = [
        Node(
            node_id="B1",
            source_id=SRC_B,
            label="Alicia Johnson",
            category="Person",
            aliases=["Alice Johnson", "A. Johnson"],
            properties={
                "employee_id": [_pv("EMP-1001", "2025-09-01T00:00:00+00:00", 0.98, "support_tickets.parquet")],
                "email": [_pv("a.johnson@acme.example", "2025-09-01T00:00:00+00:00", 0.96, "marketing_sync.json")],
                "role": [_pv("Senior Engineer", "2025-09-01T00:00:00+00:00", 0.96, "support_tickets.parquet")],
                "location": [_pv("Remote", "2025-09-01T00:00:00+00:00", 0.88, "marketing_sync.json")],
                "phone": [_pv("+1-555-0101", "2025-09-01T00:00:00+00:00", 0.70, "marketing_sync.json")],
            },
            provenance=_prov("support_tickets.parquet", 0.93),
        ),
        Node(
            node_id="B2",
            source_id=SRC_B,
            label="Robert Smith",
            category="Person",
            aliases=["Bob Smith"],
            properties={
                "employee_id": [_pv("EMP-1002", "2024-07-01T00:00:00+00:00", 0.97, "support_tickets.parquet")],
                "email": [_pv("bobsmith@acme.example", "2024-07-01T00:00:00+00:00", 0.62, "bad_scrape.html")],
                "role": [_pv("Data Analyst", "2024-07-01T00:00:00+00:00", 0.89, "support_tickets.parquet")],
                "location": [_pv("Boston", "2024-07-01T00:00:00+00:00", 0.86, "support_tickets.parquet")],
            },
            provenance=_prov("support_tickets.parquet", 0.91),
        ),
        Node(
            node_id="B3",
            source_id=SRC_B,
            label="Carol D'Az",
            category="Person",
            aliases=["Carol Diaz"],
            properties={
                "employee_id": [_pv("EMP-1003", "2024-07-01T00:00:00+00:00", 0.98, "support_tickets.parquet")],
                "email": [_pv("carol.diaz@acme.example", "2024-07-01T00:00:00+00:00", 0.94, "marketing_sync.json")],
                "role": [_pv("Engineering Manager", "2024-07-01T00:00:00+00:00", 0.95, "support_tickets.parquet")],
                "location": [_pv("Austin", "2024-07-01T00:00:00+00:00", 0.88, "support_tickets.parquet")],
            },
            provenance=_prov("support_tickets.parquet", 0.92),
        ),
        Node(
            node_id="B4",
            source_id=SRC_B,
            label="ACME Corporation",
            category="Organization",
            aliases=["Acme Corp", "ACME"],
            properties={
                "legal_id": [_pv("ORG-ACME", "2025-01-10T00:00:00+00:00", 0.99, "vendor_master.csv")],
                "legal_name": [_pv("ACME Corporation", "2025-01-10T00:00:00+00:00", 0.96, "vendor_master.csv")],
                "industry": [
                    _pv("Software", "2025-01-10T00:00:00+00:00", 0.93, "vendor_master.csv"),
                    _pv("Cloud", "2025-01-10T00:00:00+00:00", 0.81, "news_articles.jsonl"),
                ],
                "location": [_pv("New York", "2025-01-10T00:00:00+00:00", 0.90, "vendor_master.csv")],
            },
            provenance=_prov("vendor_master.csv", 0.95),
        ),
        Node(
            node_id="B5",
            source_id=SRC_B,
            label="Initech",
            category="Organization",
            aliases=[],
            properties={
                "legal_id": [_pv("ORG-INITECH", "2025-01-10T00:00:00+00:00", 0.97, "vendor_master.csv")],
                "legal_name": [_pv("Initech", "2025-01-10T00:00:00+00:00", 0.94, "vendor_master.csv")],
                "industry": [_pv("Consulting", "2025-01-10T00:00:00+00:00", 0.90, "vendor_master.csv")],
            },
            provenance=_prov("vendor_master.csv", 0.95),
        ),
        Node(
            node_id="B6",
            source_id=SRC_B,
            label="Dave Lee",
            category="Person",
            aliases=[],
            properties={
                "email": [_pv("dave.lee@initech.example", "2025-01-10T00:00:00+00:00", 0.88, "vendor_contacts.csv")],
                "role": [_pv("Consultant", "2025-01-10T00:00:00+00:00", 0.86, "vendor_contacts.csv")],
                "location": [_pv("Chicago", "2025-01-10T00:00:00+00:00", 0.82, "vendor_contacts.csv")],
            },
            provenance=_prov("vendor_contacts.csv", 0.89),
        ),
    ]

    edges_b = [
        Edge(
            edge_id="EB1",
            source_id=SRC_B,
            source_node_id="B1",
            target_node_id="B4",
            relation="works_at",
            provenance=_prov("support_tickets.parquet", 0.94),
        ),
        Edge(
            edge_id="EB2",
            source_id=SRC_B,
            source_node_id="B2",
            target_node_id="B4",
            relation="works_at",
            provenance=_prov("support_tickets.parquet", 0.94),
        ),
        Edge(
            edge_id="EB3",
            source_id=SRC_B,
            source_node_id="B3",
            target_node_id="B1",
            relation="manages",
            provenance=_prov("org_chart.json", 0.92),
        ),
        Edge(
            edge_id="EB4",
            source_id=SRC_B,
            source_node_id="B4",
            target_node_id="B5",
            relation="partner",
            provenance=_prov("vendor_master.csv", 0.88),
        ),
        Edge(
            edge_id="EB5",
            source_id=SRC_B,
            source_node_id="B6",
            target_node_id="B5",
            relation="works_at",
            provenance=_prov("vendor_contacts.csv", 0.90),
        ),
    ]

    graph_a = GraphInput(source_id=SRC_A, nodes=nodes_a, edges=edges_a)
    graph_b = GraphInput(source_id=SRC_B, nodes=nodes_b, edges=edges_b)
    return graph_a, graph_b


GOLD_DEMO_PAIRS: List[Tuple[str, str, str, str]] = [
    (SRC_A, "A1", SRC_B, "B1"),
    (SRC_A, "A2", SRC_B, "B2"),
    (SRC_A, "A3", SRC_B, "B3"),
    (SRC_A, "A4", SRC_B, "B4"),
]

GOLD_EXPECTED_RESOLUTIONS: Dict[Tuple[str, str, str, str, str], str] = {
    (SRC_A, "A1", SRC_B, "B1", "email"): "a.johnson@acme.example",
    (SRC_A, "A1", SRC_B, "B1", "role"): "Senior Engineer",
    (SRC_A, "A2", SRC_B, "B2", "email"): "bob.smith@acme.example",
}

PROPERTY_POLICY: Dict[str, str] = {
    "employee_id": "functional",
    "legal_id": "functional",
    "email": "functional",
    "role": "functional",
    "phone": "functional",
    "legal_name": "functional",
    "location": "multi_valued",
    "industry": "multi_valued",
    "skills": "multi_valued",
}

ALLOWED_RELATIONS = {"works_at", "manages", "partner", "competitor", "located_in"}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


def default_config() -> Dict[str, Any]:
    return {
        "seed": DEFAULT_SEED,
        "auto_threshold": 0.82,
        "review_threshold": 0.58,
        "weights": {
            "id": 0.35,
            "label": 0.20,
            "alias": 0.20,
            "embedding": 0.15,
            "neighborhood": 0.10,
        },
        "source_reliability": {
            SRC_A: 0.78,
            SRC_B: 0.82,
        },
        "weighted_margin": 0.05,
        "temporal_days": 180,
        "confidence_gap": 0.25,
    }


def normalize_cfg(raw: Any) -> Dict[str, Any]:
    cfg = default_config()

    if not isinstance(raw, dict):
        return cfg

    try:
        cfg["seed"] = int(raw.get("seed", cfg["seed"]))
    except Exception:
        pass

    for key in ("auto_threshold", "review_threshold", "weighted_margin", "confidence_gap"):
        try:
            cfg[key] = float(raw.get(key, cfg[key]))
        except Exception:
            pass

    try:
        cfg["temporal_days"] = int(raw.get("temporal_days", cfg["temporal_days"]))
    except Exception:
        pass

    weights = raw.get("weights")
    if isinstance(weights, dict):
        for key in cfg["weights"]:
            try:
                cfg["weights"][key] = float(weights.get(key, cfg["weights"][key]))
            except Exception:
                pass

    reliability = raw.get("source_reliability")
    if isinstance(reliability, dict):
        for key in (SRC_A, SRC_B):
            try:
                cfg["source_reliability"][key] = float(
                    reliability.get(key, cfg["source_reliability"][key])
                )
            except Exception:
                pass

    if cfg["review_threshold"] > cfg["auto_threshold"]:
        cfg["review_threshold"] = cfg["auto_threshold"]

    return cfg


# -----------------------------------------------------------------------------
# Core reconciliation logic
# -----------------------------------------------------------------------------


def generate_candidates(
    graph_a: GraphInput,
    graph_b: GraphInput,
    cfg: Dict[str, Any],
    logs: List[Dict[str, Any]],
) -> List[MatchCandidate]:
    g_a = build_nx_graph(graph_a)
    g_b = build_nx_graph(graph_b)
    emb_cache: Dict[str, List[float]] = {}

    def get_emb(text: str) -> List[float]:
        key = normalize_text(text)
        if key not in emb_cache:
            emb_cache[key] = pseudo_embedding(text)
        return emb_cache[key]

    weights = cfg["weights"]
    total_weight = sum(float(v) for v in weights.values()) or 1.0
    candidates: List[MatchCandidate] = []

    for na in graph_a.nodes:
        for nb in graph_b.nodes:
            if na.category != nb.category:
                continue

            ida = latest_claim(na, "employee_id") or latest_claim(na, "legal_id")
            idb = latest_claim(nb, "employee_id") or latest_claim(nb, "legal_id")

            id_score = 0.0
            if ida is not None and idb is not None:
                if normalize_value("employee_id", ida.value) == normalize_value("employee_id", idb.value):
                    id_score = 1.0

            label_score = SequenceMatcher(
                None,
                normalize_text(na.label),
                normalize_text(nb.label),
            ).ratio()

            terms_a = [na.label] + na.aliases
            terms_b = [nb.label] + nb.aliases
            alias_score = 0.0

            for ta in terms_a:
                for tb in terms_b:
                    alias_score = max(
                        alias_score,
                        SequenceMatcher(None, normalize_text(ta), normalize_text(tb)).ratio(),
                    )

            text_a = node_text(na)
            text_b = node_text(nb)
            embedding_score = cosine_similarity(get_emb(text_a), get_emb(text_b))

            sig_a = neighbor_signature(g_a, uid(graph_a.source_id, na.node_id))
            sig_b = neighbor_signature(g_b, uid(graph_b.source_id, nb.node_id))
            neighborhood_score = jaccard_similarity(sig_a, sig_b)

            score = (
                float(weights["id"]) * id_score
                + float(weights["label"]) * label_score
                + float(weights["alias"]) * alias_score
                + float(weights["embedding"]) * embedding_score
                + float(weights["neighborhood"]) * neighborhood_score
            ) / total_weight

            score = max(0.0, min(1.0, score))

            if score >= float(cfg["auto_threshold"]):
                band = "auto_accept"
            elif score >= float(cfg["review_threshold"]):
                band = "review"
            else:
                band = "reject"

            evidence: List[str] = []

            if id_score == 1.0:
                evidence.append("exact_identifier_match")
            if label_score >= 0.85:
                evidence.append("high_label_similarity")
            if alias_score >= 0.95:
                evidence.append("alias_match")
            if embedding_score >= 0.75:
                evidence.append("local_surrogate_embedding_similarity")
            if neighborhood_score > 0.0:
                evidence.append("neighbourhood_overlap")

            candidates.append(
                MatchCandidate(
                    left_source=graph_a.source_id,
                    left_id=na.node_id,
                    right_source=graph_b.source_id,
                    right_id=nb.node_id,
                    left_label=na.label,
                    right_label=nb.label,
                    score=score,
                    id_score=id_score,
                    label_score=label_score,
                    alias_score=alias_score,
                    embedding_score=embedding_score,
                    neighborhood_score=neighborhood_score,
                    band=band,
                    evidence=evidence,
                )
            )

    candidates.sort(key=lambda m: m.score, reverse=True)
    log_event(logs, "INFO", "candidates_generated", count=len(candidates))
    return candidates


def classify_functional_conflict(
    claim_a: PropertyValue,
    claim_b: PropertyValue,
    property_name: str,
    cfg: Dict[str, Any],
) -> Literal["error", "temporal", "perspective"]:
    valid_a = validate_property_value(property_name, claim_a.value)
    valid_b = validate_property_value(property_name, claim_b.value)

    if not valid_a or not valid_b:
        return "error"

    delta_days = abs((claim_a.observed_at - claim_b.observed_at).days)
    if delta_days >= int(cfg["temporal_days"]):
        return "temporal"

    if abs(claim_a.confidence - claim_b.confidence) >= float(cfg["confidence_gap"]):
        return "error"

    return "perspective"


def make_claim(
    source_id: str,
    node_id: str,
    property_name: str,
    pv: PropertyValue,
    cfg: Dict[str, Any],
) -> Claim:
    reliability = float(cfg["source_reliability"].get(source_id, 0.75))
    weight = reliability * float(pv.confidence)

    if not validate_property_value(property_name, pv.value):
        weight *= 0.2

    return Claim(
        source_id=source_id,
        node_id=node_id,
        property_name=property_name,
        value=pv.value,
        observed_at=pv.observed_at,
        confidence=pv.confidence,
        reliability=reliability,
        weight=weight,
        provenance=pv.provenance,
    )


def resolve_conflict(
    pair_id: str,
    left_source: str,
    left_id: str,
    right_source: str,
    right_id: str,
    property_name: str,
    claims: List[Claim],
    conflict_type: Literal["error", "temporal", "perspective"],
    functional: bool,
    cfg: Dict[str, Any],
) -> ConflictRecord:
    claims_sorted = sorted(claims, key=lambda c: (c.weight, c.observed_at), reverse=True)
    valid_claims = [c for c in claims_sorted if validate_property_value(property_name, c.value)]

    weighted_winner: Optional[str] = None
    majority_winner: Optional[str] = None
    status: Literal["resolved", "review"] = "review"
    rationale = ""

    if not functional:
        rationale = "Multi-valued property: competing claims are retained and routed to review."
        return ConflictRecord(
            pair_id=pair_id,
            left_source=left_source,
            left_id=left_id,
            right_source=right_source,
            right_id=right_id,
            property_name=property_name,
            conflict_type=conflict_type,
            functional=functional,
            claims=claims_sorted,
            weighted_winner=None,
            majority_winner=None,
            agreement=False,
            status="review",
            rationale=rationale,
        )

    if not valid_claims:
        rationale = "No claim passed format validation; human review required."
    elif len(valid_claims) == 1 and len(claims_sorted) > 1:
        weighted_winner = valid_claims[0].value
        majority_winner = weighted_winner
        status = "resolved"
        rationale = "Only one format-valid claim remained after downweighting invalid competitors."
    else:
        top = valid_claims[0]
        second = valid_claims[1] if len(valid_claims) > 1 else None
        margin = top.weight - (second.weight if second is not None else top.weight)

        norm_to_raw: Dict[str, str] = {}
        counts: Counter[str] = Counter()

        for c in valid_claims:
            nv = normalize_value(property_name, c.value)
            norm_to_raw.setdefault(nv, c.value)
            counts[nv] += 1

        max_count = max(counts.values())
        majority_keys = [k for k, v in counts.items() if v == max_count]
        majority_winner = norm_to_raw[majority_keys[0]] if len(majority_keys) == 1 else None

        if second is None or margin >= float(cfg["weighted_margin"]):
            weighted_winner = top.value
            status = "resolved"
            rationale = (
                f"Weighted source-reliability margin {margin:.3f} "
                f">= threshold {float(cfg['weighted_margin']):.3f}."
            )
        else:
            status = "review"
            rationale = (
                f"Weighted margin {margin:.3f} below threshold "
                f"{float(cfg['weighted_margin']):.3f}; routed to human review."
            )

    agreement = (
        weighted_winner is not None
        and majority_winner is not None
        and normalize_value(property_name, weighted_winner) == normalize_value(property_name, majority_winner)
    )

    return ConflictRecord(
        pair_id=pair_id,
        left_source=left_source,
        left_id=left_id,
        right_source=right_source,
        right_id=right_id,
        property_name=property_name,
        conflict_type=conflict_type,
        functional=functional,
        claims=claims_sorted,
        weighted_winner=weighted_winner,
        majority_winner=majority_winner,
        agreement=agreement,
        status=status,
        rationale=rationale,
    )


def detect_conflicts(
    matches: List[MatchCandidate],
    graph_a: GraphInput,
    graph_b: GraphInput,
    cfg: Dict[str, Any],
    logs: List[Dict[str, Any]],
) -> List[ConflictRecord]:
    nodes_a = {n.node_id: n for n in graph_a.nodes}
    nodes_b = {n.node_id: n for n in graph_b.nodes}
    conflicts: List[ConflictRecord] = []

    for m in matches:
        if m.band == "reject":
            continue

        na = nodes_a.get(m.left_id)
        nb = nodes_b.get(m.right_id)

        if na is None or nb is None:
            continue

        pair_id = f"{m.left_source}:{m.left_id}|{m.right_source}:{m.right_id}"
        property_names = set(na.properties.keys()) | set(nb.properties.keys())

        for prop in sorted(property_names):
            policy = PROPERTY_POLICY.get(prop, "functional")

            if policy == "functional":
                ca = latest_claim(na, prop)
                cb = latest_claim(nb, prop)

                if ca is None or cb is None:
                    continue

                if normalize_value(prop, ca.value) == normalize_value(prop, cb.value):
                    continue

                conflict_type = classify_functional_conflict(ca, cb, prop, cfg)

                claims = [
                    make_claim(graph_a.source_id, na.node_id, prop, ca, cfg),
                    make_claim(graph_b.source_id, nb.node_id, prop, cb, cfg),
                ]

                conflicts.append(
                    resolve_conflict(
                        pair_id=pair_id,
                        left_source=m.left_source,
                        left_id=m.left_id,
                        right_source=m.right_source,
                        right_id=m.right_id,
                        property_name=prop,
                        claims=claims,
                        conflict_type=conflict_type,
                        functional=True,
                        cfg=cfg,
                    )
                )

            else:
                claims_a = all_unique_claims(na, prop)
                claims_b = all_unique_claims(nb, prop)

                map_a = {normalize_value(prop, c.value): c for c in claims_a}
                map_b = {normalize_value(prop, c.value): c for c in claims_b}

                symmetric = set(map_a.keys()) ^ set(map_b.keys())
                if not symmetric:
                    continue

                claims: List[Claim] = []

                for key in symmetric:
                    if key in map_a:
                        claims.append(make_claim(graph_a.source_id, na.node_id, prop, map_a[key], cfg))
                    if key in map_b:
                        claims.append(make_claim(graph_b.source_id, nb.node_id, prop, map_b[key], cfg))

                if len(claims) < 2:
                    continue

                conflict_type_multi: Literal["error", "temporal", "perspective"] = "perspective"
                if any(not validate_property_value(prop, c.value) for c in claims):
                    conflict_type_multi = "error"

                conflicts.append(
                    resolve_conflict(
                        pair_id=pair_id,
                        left_source=m.left_source,
                        left_id=m.left_id,
                        right_source=m.right_source,
                        right_id=m.right_id,
                        property_name=prop,
                        claims=claims,
                        conflict_type=conflict_type_multi,
                        functional=False,
                        cfg=cfg,
                    )
                )

    log_event(logs, "INFO", "conflicts_detected", count=len(conflicts))
    return conflicts


# -----------------------------------------------------------------------------
# Evaluation and self-checks
# -----------------------------------------------------------------------------


def validate_shapes(graph_a: GraphInput, graph_b: GraphInput) -> Dict[str, Any]:
    checks = 0
    violations = 0

    for g in (graph_a, graph_b):
        for n in g.nodes:
            checks += 3

            if not n.label.strip():
                violations += 1
            if not n.category.strip():
                violations += 1
            if not n.provenance.source_document.strip():
                violations += 1

            for prop, claims in n.properties.items():
                checks += 1
                if not prop.strip():
                    violations += 1

                for c in claims:
                    checks += 1
                    if not c.value.strip():
                        violations += 1

        for e in g.edges:
            checks += 2

            if e.relation not in ALLOWED_RELATIONS:
                violations += 1
            if not e.provenance.source_document.strip():
                violations += 1

    conformance = 1.0 if checks == 0 else max(0.0, 1.0 - (violations / checks))

    return {
        "checks": checks,
        "violations": violations,
        "schema_conformance": conformance,
        "note": (
            "Lightweight Pydantic/shape checks. Full SHACL validation is TO VERIFY "
            "and not implemented in this single-file prototype."
        ),
    }


def compute_metrics(
    matches: List[MatchCandidate],
    conflicts: List[ConflictRecord],
    graph_a: GraphInput,
    graph_b: GraphInput,
    cfg: Dict[str, Any],
    is_demo: bool,
) -> Dict[str, Any]:
    shape_metrics = validate_shapes(graph_a, graph_b)

    alignment: Dict[str, Any]

    if is_demo and GOLD_DEMO_PAIRS:
        predicted_auto = {
            (m.left_source, m.left_id, m.right_source, m.right_id)
            for m in matches
            if m.band == "auto_accept"
        }
        predicted_review = {
            (m.left_source, m.left_id, m.right_source, m.right_id)
            for m in matches
            if m.band == "review"
        }
        gold = set(GOLD_DEMO_PAIRS)

        tp = len(predicted_auto & gold)
        fp = len(predicted_auto - gold)
        fn = len(gold - predicted_auto)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        coverage = len((predicted_auto | predicted_review) & gold) / len(gold) if gold else 0.0

        alignment = {
            "gold_pairs": len(gold),
            "auto_precision": precision,
            "auto_recall": recall,
            "auto_f1": f1,
            "auto_plus_review_coverage": coverage,
            "incorrectly_collapsed_count": fp,
            "incorrectly_collapsed": [list(x) for x in sorted(predicted_auto - gold)],
            "note": "Synthetic demo gold labels only. Not a real-world benchmark.",
        }
    else:
        alignment = {
            "note": (
                "No gold labels available for uploaded graphs. "
                "Entity-resolution precision/recall are TO VERIFY."
            ),
            "candidate_count": len(matches),
            "auto_count": sum(1 for m in matches if m.band == "auto_accept"),
            "review_count": sum(1 for m in matches if m.band == "review"),
        }

    resolved_conflicts = [c for c in conflicts if c.status == "resolved"]
    review_conflicts = [c for c in conflicts if c.status == "review"]

    retained_claims = all(len(c.claims) >= 2 for c in conflicts)
    provenance_complete = all(
        bool(claim.provenance.source_document.strip())
        for c in conflicts
        for claim in c.claims
    )

    labelled_total = 0
    td_correct = 0
    majority_correct = 0

    if is_demo:
        for c in conflicts:
            key = (c.left_source, c.left_id, c.right_source, c.right_id, c.property_name)
            expected = GOLD_EXPECTED_RESOLUTIONS.get(key)

            if expected is None:
                continue

            labelled_total += 1

            if c.status == "resolved" and c.weighted_winner is not None:
                if normalize_value(c.property_name, c.weighted_winner) == normalize_value(c.property_name, expected):
                    td_correct += 1

            if c.status == "resolved" and c.majority_winner is not None:
                if normalize_value(c.property_name, c.majority_winner) == normalize_value(c.property_name, expected):
                    majority_correct += 1

    conflict_metrics = {
        "conflict_count": len(conflicts),
        "resolved_count": len(resolved_conflicts),
        "review_count": len(review_conflicts),
        "all_conflicting_claims_retained": retained_claims,
        "provenance_complete": provenance_complete,
        "graph_consistency_score": 1.0 if retained_claims and provenance_complete else 0.0,
        "labelled_conflict_count": labelled_total,
        "truth_discovery_accuracy": (td_correct / labelled_total) if labelled_total else 0.0,
        "majority_vote_accuracy": (majority_correct / labelled_total) if labelled_total else 0.0,
        "note": (
            "Labelled conflict metrics are computed only on the synthetic demo "
            "gold resolution subset."
        ),
    }

    error_analysis: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for c in conflicts:
        error_analysis[c.property_name][c.conflict_type] += 1
        error_analysis[c.property_name][f"status_{c.status}"] += 1

    return {
        "alignment": alignment,
        "conflicts": conflict_metrics,
        "shapes": shape_metrics,
        "error_analysis_by_property": {k: dict(v) for k, v in error_analysis.items()},
        "not_implemented_or_to_verify": {
            "relation_extraction_f1": (
                "TO VERIFY: relation extraction is not trained/evaluated "
                "in this single-file prototype."
            ),
            "shacl_conformance": (
                "TO VERIFY: lightweight shape checks are used instead of "
                "full pySHACL validation."
            ),
            "citation_faithfulness": (
                "TO VERIFY: no LLM citations are generated in this offline prototype."
            ),
            "retrieval_ndcg_mrr_recall_at_k": (
                "TO VERIFY: no vector retrieval index is used in this single-file prototype."
            ),
            "recovery_time": (
                "TO VERIFY: failure recovery is demonstrated only by graceful "
                "error handling, not full job resume."
            ),
        },
    }


def run_self_checks(
    cfg: Dict[str, Any],
    graph_a: GraphInput,
    graph_b: GraphInput,
    conflicts: List[ConflictRecord],
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append({"check": name, "passed": bool(passed), "detail": detail})

    try:
        GraphInput.model_validate(
            {
                "schema_version": SCHEMA_VERSION,
                "source_id": "X",
                "nodes": [],
                "edges": [],
                "unexpected_field": 1,
            }
        )
        check("unknown_field_rejected", False, "Validation unexpectedly passed.")
    except ValidationError:
        check("unknown_field_rejected", True, "Extra fields are rejected at trust boundary.")

    check(
        "deterministic_fingerprint",
        fingerprint_graph(graph_a) == fingerprint_graph(graph_a),
        "SHA-256 fingerprint is stable.",
    )

    check(
        "normalization_unicode",
        normalize_text("Carol D'Az") == "carol d'az",
        "Accent/punctuation normalization works.",
    )

    check(
        "pseudo_embedding_deterministic",
        pseudo_embedding("Alice Johnson") == pseudo_embedding("Alice Johnson"),
        "Local surrogate embedding is deterministic.",
    )

    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "atomic_test.txt"
            atomic_write_bytes(p, b"ok")
            check("atomic_write", p.read_bytes() == b"ok", "Temporary file commit works.")
    except Exception as exc:
        check("atomic_write", False, str(exc))

    check(
        "conflict_claims_retained",
        all(len(c.claims) >= 2 for c in conflicts),
        f"{len(conflicts)} conflicts inspected.",
    )

    check(
        "seed_config_present",
        isinstance(cfg["seed"], int),
        "Explicit deterministic seed is configured.",
    )

    return results


# -----------------------------------------------------------------------------
# Animation renderer
# -----------------------------------------------------------------------------


def get_font(size: int) -> ImageFont.ImageFont:
    for name in ["arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x - w / 2, y - h / 2), text, font=font, fill=fill)


def render_animation(
    graph_a: GraphInput,
    graph_b: GraphInput,
    matches: List[MatchCandidate],
    conflicts: List[ConflictRecord],
    cfg: Dict[str, Any],
) -> Tuple[bytes, bytes]:
    W, H = 980, 580
    margin = 78
    total_frames = 66

    G = nx.Graph()

    for n in graph_a.nodes:
        G.add_node(
            uid(graph_a.source_id, n.node_id),
            label=n.label,
            source=graph_a.source_id,
            category=n.category,
        )

    for n in graph_b.nodes:
        G.add_node(
            uid(graph_b.source_id, n.node_id),
            label=n.label,
            source=graph_b.source_id,
            category=n.category,
        )

    for e in graph_a.edges:
        G.add_edge(
            uid(e.source_id, e.source_node_id),
            uid(e.source_id, e.target_node_id),
            relation=e.relation,
        )

    for e in graph_b.edges:
        G.add_edge(
            uid(e.source_id, e.source_node_id),
            uid(e.source_id, e.target_node_id),
            relation=e.relation,
        )

    positions: Dict[str, Tuple[int, int]] = {}

    if G.number_of_nodes() > 0:
        raw = nx.spring_layout(G, seed=int(cfg["seed"]), k=0.95)
        xs = [float(p[0]) for p in raw.values()]
        ys = [float(p[1]) for p in raw.values()]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        span_x = (max_x - min_x) or 1.0
        span_y = (max_y - min_y) or 1.0

        for k, v in raw.items():
            x = int(margin + ((float(v[0]) - min_x) / span_x) * (W - 2 * margin))
            y = int(margin + ((float(v[1]) - min_y) / span_y) * (H - 2 * margin - 55))
            positions[str(k)] = (x, y)

    auto_ids: set[str] = set()
    review_ids: set[str] = set()

    for m in matches:
        lu = uid(m.left_source, m.left_id)
        ru = uid(m.right_source, m.right_id)

        if m.band == "auto_accept":
            auto_ids.add(lu)
            auto_ids.add(ru)
        elif m.band == "review":
            review_ids.add(lu)
            review_ids.add(ru)

    conflict_ids: set[str] = set()

    for c in conflicts:
        lu = uid(c.left_source, c.left_id)
        ru = uid(c.right_source, c.right_id)
        conflict_ids.add(lu)
        conflict_ids.add(ru)

    sorted_matches = sorted(matches, key=lambda m: m.score, reverse=True)

    font_title = get_font(20)
    font_small = get_font(13)
    font_tiny = get_font(11)

    frames: List[Image.Image] = []

    for frame in range(total_frames):
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)

        if frame < 16:
            phase = "1. Loading source graphs and provenance"
        elif frame < 38:
            phase = "2. Scoring candidate entity alignments"
        elif frame < 52:
            phase = "3. Detecting property conflicts"
        else:
            phase = "4. Ranking claims and routing review"

        d.rectangle([0, 0, W, 46], fill="#111827")
        d.text((16, 12), "Knowledge Graph Data Reconciliation - Live Demo", font=font_title, fill="white")
        d.text((16, H - 34), f"{phase} | frame {frame + 1}/{total_frames}", font=font_small, fill="#374151")

        for u, v in G.edges():
            if u in positions and v in positions:
                d.line([positions[u], positions[v]], fill="#cbd5e1", width=1)

        if frame >= 16:
            progress = min(1.0, (frame - 16) / 22.0)
            reveal_count = int(progress * len(sorted_matches))

            for m in sorted_matches[:reveal_count]:
                if m.band == "reject":
                    continue

                lu = uid(m.left_source, m.left_id)
                ru = uid(m.right_source, m.right_id)

                if lu in positions and ru in positions:
                    color = "#16a34a" if m.band == "auto_accept" else "#f59e0b"
                    d.line([positions[lu], positions[ru]], fill=color, width=2)

        for node_uid, pos in positions.items():
            source = G.nodes[node_uid].get("source", SRC_A)
            label = str(G.nodes[node_uid].get("label", node_uid))

            base_color = "#2563eb" if source == SRC_A else "#f97316"
            radius = 17

            if frame >= 16:
                if node_uid in auto_ids:
                    base_color = "#16a34a"
                elif node_uid in review_ids:
                    base_color = "#f59e0b"

            pulse = 0
            if frame >= 38 and node_uid in conflict_ids:
                base_color = "#dc2626"
                pulse = int(3 * abs(math.sin(frame * 0.45)))

            xi, yi = pos
            r = radius + pulse

            d.ellipse(
                [xi - r, yi - r, xi + r, yi + r],
                fill=base_color,
                outline="#111827",
                width=1,
            )

            short = label if len(label) <= 16 else label[:15] + "..."
            centered_text(d, xi, yi + r + 12, short, font_tiny, "#111827")

        if frame >= 52:
            for c in conflicts:
                lu = uid(c.left_source, c.left_id)
                ru = uid(c.right_source, c.right_id)

                if lu in positions and ru in positions:
                    mx = int((positions[lu][0] + positions[ru][0]) / 2)
                    my = int((positions[lu][1] + positions[ru][1]) / 2)

                    color = "#16a34a" if c.status == "resolved" else "#f59e0b"
                    symbol = "OK" if c.status == "resolved" else "?"

                    d.ellipse(
                        [mx - 14, my - 14, mx + 14, my + 14],
                        fill=color,
                        outline="white",
                        width=2,
                    )
                    centered_text(d, mx, my, symbol, font_tiny, "white")

        lx, ly = W - 235, 58

        d.rectangle(
            [lx - 10, ly - 10, lx + 220, ly + 112],
            outline="#9ca3af",
            width=1,
            fill="white",
        )

        d.text((lx, ly), "Legend", font=font_small, fill="#111827")

        legend = [
            ("#2563eb", "Source A node"),
            ("#f97316", "Source B node"),
            ("#16a34a", "Auto match / resolved"),
            ("#f59e0b", "Review band"),
            ("#dc2626", "Conflict highlighted"),
        ]

        for i, (color, text) in enumerate(legend):
            yy = ly + 24 + i * 17
            d.ellipse([lx, yy - 5, lx + 10, yy + 5], fill=color)
            d.text((lx + 18, yy - 7), text, font=font_tiny, fill="#111827")

        frames.append(img)

    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=90,
        loop=0,
        optimize=False,
    )
    gif_buf.seek(0)
    gif_bytes = gif_buf.read()

    png_buf = io.BytesIO()
    frames[-1].save(png_buf, format="PNG")
    png_buf.seek(0)
    png_bytes = png_buf.read()

    return gif_bytes, png_bytes


# -----------------------------------------------------------------------------
# Pipeline runner
# -----------------------------------------------------------------------------


def save_artifacts(
    result_json: Dict[str, Any],
    gif_bytes: bytes,
    png_bytes: bytes,
    cfg_h: str,
    logs: List[Dict[str, Any]],
) -> Optional[str]:
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe_hash = re.sub(r"[^a-zA-Z0-9_-]", "", cfg_h)[:16]
        base = ARTIFACT_DIR / f"run_{stamp}_{safe_hash}"

        atomic_write_bytes(
            base.with_suffix(".json"),
            json.dumps(result_json, indent=2, sort_keys=True).encode("utf-8"),
        )
        atomic_write_bytes(base.with_suffix(".gif"), gif_bytes)
        atomic_write_bytes(base.with_suffix(".png"), png_bytes)

        log_event(logs, "INFO", "artifacts_saved", path=str(base))
        return str(base)
    except OSError as exc:
        log_event(logs, "WARNING", "artifact_save_failed", error=str(exc))
        return None


def run_pipeline(
    cfg: Dict[str, Any],
    graph_a: Optional[GraphInput] = None,
    graph_b: Optional[GraphInput] = None,
    inject_failure: bool = False,
) -> Dict[str, Any]:
    logs: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    cfg_h = ""
    tracemalloc.start()

    try:
        effective_cfg = dict(cfg)
        effective_cfg["inject_failure"] = bool(inject_failure)
        cfg_h = config_hash(effective_cfg)

        random.seed(int(effective_cfg["seed"]))
        log_event(
            logs,
            "INFO",
            "pipeline_started",
            seed=effective_cfg["seed"],
            app_version=APP_VERSION,
            config_hash=cfg_h[:16],
            inject_failure=bool(inject_failure),
        )

        if inject_failure:
            raise AppError(
                ERROR_CODES["invalid_value"],
                "Synthetic corrupted-input failure injected for robustness drill.",
            )

        if graph_a is None or graph_b is None:
            ga, gb = make_demo_graphs()
            is_demo = True
            log_event(
                logs,
                "INFO",
                "demo_data_loaded",
                nodes_a=len(ga.nodes),
                nodes_b=len(gb.nodes),
            )
        else:
            ga, gb = graph_a, graph_b
            is_demo = False
            log_event(
                logs,
                "INFO",
                "uploaded_graphs_loaded",
                nodes_a=len(ga.nodes),
                nodes_b=len(gb.nodes),
            )

        fp_a = fingerprint_graph(ga)
        fp_b = fingerprint_graph(gb)

        log_event(
            logs,
            "INFO",
            "inputs_fingerprinted",
            source_a=fp_a[:16],
            source_b=fp_b[:16],
            config_hash=cfg_h[:16],
        )

        matches = generate_candidates(ga, gb, effective_cfg, logs)
        conflicts = detect_conflicts(matches, ga, gb, effective_cfg, logs)
        metrics = compute_metrics(matches, conflicts, ga, gb, effective_cfg, is_demo)
        self_checks = run_self_checks(effective_cfg, ga, gb, conflicts)

        gif_bytes, png_bytes = render_animation(ga, gb, matches, conflicts, effective_cfg)

        elapsed = time.perf_counter() - start_time
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        total_nodes = len(ga.nodes) + len(gb.nodes)
        throughput = total_nodes / elapsed if elapsed > 0 else 0.0

        runtime = {
            "elapsed_seconds": elapsed,
            "peak_memory_bytes": peak_mem,
            "peak_memory_mb": peak_mem / (1024 * 1024),
            "throughput_nodes_per_second": throughput,
            "failure_rate_this_run": 0.0,
            "recovery_time": "TO VERIFY: not exercised in happy-path run.",
        }

        result_json = {
            "app_version": APP_VERSION,
            "schema_version": SCHEMA_VERSION,
            "config": effective_cfg,
            "config_hash": cfg_h,
            "input_fingerprints": {
                "graph_a_sha256": fp_a,
                "graph_b_sha256": fp_b,
            },
            "matches": [m.model_dump(mode="json") for m in matches],
            "conflicts": [c.model_dump(mode="json") for c in conflicts],
            "metrics": metrics,
            "self_checks": self_checks,
            "logs": logs,
            "runtime": runtime,
            "disclaimer": "Synthetic demo metrics only. Not a real-world validation claim.",
        }

        artifact_path = save_artifacts(result_json, gif_bytes, png_bytes, cfg_h, logs)

        return {
            "config": effective_cfg,
            "config_hash": cfg_h,
            "graph_a": ga,
            "graph_b": gb,
            "matches": matches,
            "conflicts": conflicts,
            "metrics": metrics,
            "self_checks": self_checks,
            "logs": logs,
            "runtime": runtime,
            "gif_bytes": gif_bytes,
            "png_bytes": png_bytes,
            "result_json": result_json,
            "artifact_path": artifact_path,
            "is_demo": is_demo,
        }

    except AppError as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        log_event(
            logs,
            "ERROR",
            "pipeline_failed",
            code=exc.code,
            message=str(exc),
            details=exc.details,
        )

        return {
            "error": {
                "code": exc.code,
                "message": str(exc),
                "details": exc.details,
                "config_hash": cfg_h,
            },
            "logs": logs,
        }

    except Exception as exc:
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        log_event(
            logs,
            "ERROR",
            "unexpected_pipeline_failure",
            exception=type(exc).__name__,
            message=str(exc),
        )

        return {
            "error": {
                "code": ERROR_CODES["unexpected"],
                "message": (
                    "Unexpected pipeline failure. See logs for sanitized "
                    "exception type and message."
                ),
                "details": {
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
                "config_hash": cfg_h,
            },
            "logs": logs,
        }


def safe_run_pipeline(
    cfg: Dict[str, Any],
    graph_a: Optional[GraphInput] = None,
    graph_b: Optional[GraphInput] = None,
    inject_failure: bool = False,
) -> Dict[str, Any]:
    return run_pipeline(cfg, graph_a, graph_b, inject_failure=inject_failure)


# -----------------------------------------------------------------------------
# Upload parsing
# -----------------------------------------------------------------------------


def parse_uploaded_graph(file: Any, label: str) -> GraphInput:
    try:
        if hasattr(file, "getvalue"):
            raw = file.getvalue()
        else:
            raw = file.read()

        if isinstance(raw, str):
            text = raw
        else:
            text = raw.decode("utf-8")

        data = json.loads(text)
        return GraphInput.model_validate(data)

    except ValidationError as exc:
        raise AppError(
            ERROR_CODES["validation_failed"],
            f"{label} failed schema validation.",
            {"error": str(exc)},
        ) from exc

    except UnicodeDecodeError as exc:
        raise AppError(
            ERROR_CODES["invalid_value"],
            f"{label} is not valid UTF-8.",
            {"error": str(exc)},
        ) from exc

    except json.JSONDecodeError as exc:
        raise AppError(
            ERROR_CODES["invalid_value"],
            f"{label} is not valid JSON.",
            {"error": str(exc)},
        ) from exc

    except Exception as exc:
        raise AppError(
            ERROR_CODES["unexpected"],
            f"{label} could not be parsed.",
            {"error": str(exc)},
        ) from exc


# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------


st.set_page_config(
    page_title="Knowledge Graph Data Reconciliation",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "cfg" not in st.session_state:
    st.session_state.cfg = default_config()
else:
    st.session_state.cfg = normalize_cfg(st.session_state.cfg)

if "failure_drill" not in st.session_state:
    st.session_state.failure_drill = None

if "pipeline" not in st.session_state or "error" in st.session_state.pipeline:
    st.session_state.pipeline = safe_run_pipeline(st.session_state.cfg)

st.title("🧭 Knowledge Graph Data Reconciliation System")
st.caption(
    "Single-file CPU-only Streamlit prototype. No API keys. Synthetic demo data loads automatically and "
    "generates an animated live demo GIF on first run."
)

with st.sidebar:
    st.header("Controls")

    with st.expander("Optional: upload graph JSON", expanded=False):
        file_a = st.file_uploader("Graph A JSON", type="json", key="file_a")
        file_b = st.file_uploader("Graph B JSON", type="json", key="file_b")
        use_upload = st.checkbox("Use uploaded graphs if valid", value=False)

        if st.button("Validate uploads"):
            if file_a is None or file_b is None:
                st.error("Provide both Graph A and Graph B JSON files.")
            else:
                try:
                    ga = parse_uploaded_graph(file_a, "Graph A")
                    gb = parse_uploaded_graph(file_b, "Graph B")
                    st.session_state.uploaded_graphs = (ga, gb)
                    st.success("Uploaded graphs validated.")
                except AppError as exc:
                    st.session_state.pop("uploaded_graphs", None)
                    st.error(f"{exc.code}: {exc}")

    cfg0 = st.session_state.cfg

    with st.form("controls_form"):
        st.subheader("Reconciliation settings")

        auto_threshold = st.slider(
            "Auto-accept threshold",
            0.50,
            1.00,
            float(cfg0["auto_threshold"]),
            0.01,
        )
        review_threshold = st.slider(
            "Review threshold",
            0.20,
            0.95,
            float(cfg0["review_threshold"]),
            0.01,
        )

        st.markdown("**Similarity weights**")
        w_id = st.slider("ID weight", 0.00, 1.00, float(cfg0["weights"]["id"]), 0.01)
        w_label = st.slider("Label weight", 0.00, 1.00, float(cfg0["weights"]["label"]), 0.01)
        w_alias = st.slider("Alias weight", 0.00, 1.00, float(cfg0["weights"]["alias"]), 0.01)
        w_emb = st.slider(
            "Surrogate embedding weight",
            0.00,
            1.00,
            float(cfg0["weights"]["embedding"]),
            0.01,
        )
        w_neigh = st.slider(
            "Neighbourhood weight",
            0.00,
            1.00,
            float(cfg0["weights"]["neighborhood"]),
            0.01,
        )

        st.markdown("**Source reliability and policy**")
        rel_a = st.slider(
            "Source A reliability",
            0.50,
            1.00,
            float(cfg0["source_reliability"][SRC_A]),
            0.01,
        )
        rel_b = st.slider(
            "Source B reliability",
            0.50,
            1.00,
            float(cfg0["source_reliability"][SRC_B]),
            0.01,
        )
        weighted_margin = st.slider(
            "Weighted margin threshold",
            0.00,
            0.50,
            float(cfg0["weighted_margin"]),
            0.01,
        )
        temporal_days = st.slider(
            "Temporal gap days",
            30,
            720,
            int(cfg0["temporal_days"]),
            10,
        )
        confidence_gap = st.slider(
            "Confidence gap for error",
            0.05,
            0.50,
            float(cfg0["confidence_gap"]),
            0.01,
        )

        submitted = st.form_submit_button("Apply and re-run")

    st.markdown("---")
    st.subheader("Robustness drill")
    st.caption(
        "Runs a controlled synthetic failure in a separate drill. It does not replace the working demo pipeline."
    )

    if st.button("Run failure drill", key="failure_drill_sidebar"):
        st.session_state.failure_drill = safe_run_pipeline(
            st.session_state.cfg,
            inject_failure=True,
        )

if submitted:
    if review_threshold > auto_threshold:
        review_threshold = auto_threshold

    new_cfg = {
        "seed": DEFAULT_SEED,
        "auto_threshold": float(auto_threshold),
        "review_threshold": float(review_threshold),
        "weights": {
            "id": float(w_id),
            "label": float(w_label),
            "alias": float(w_alias),
            "embedding": float(w_emb),
            "neighborhood": float(w_neigh),
        },
        "source_reliability": {
            SRC_A: float(rel_a),
            SRC_B: float(rel_b),
        },
        "weighted_margin": float(weighted_margin),
        "temporal_days": int(temporal_days),
        "confidence_gap": float(confidence_gap),
    }

    new_cfg = normalize_cfg(new_cfg)
    st.session_state.cfg = new_cfg

    if use_upload and "uploaded_graphs" in st.session_state:
        ga, gb = st.session_state.uploaded_graphs
        st.session_state.pipeline = safe_run_pipeline(new_cfg, ga, gb)
    else:
        if use_upload and "uploaded_graphs" not in st.session_state:
            st.sidebar.warning("Upload two valid graphs first; falling back to synthetic demo data.")
        st.session_state.pipeline = safe_run_pipeline(new_cfg)

pipeline = st.session_state.pipeline

if "error" in pipeline:
    st.error(f"`{pipeline['error']['code']}`: {pipeline['error']['message']}")

    with st.expander("Error details and logs", expanded=True):
        st.json(pipeline["error"])
        st.json(pipeline.get("logs", []))

    c1, c2 = st.columns(2)

    with c1:
        if st.button("Reset configuration and rerun demo"):
            st.session_state.cfg = default_config()
            st.session_state.pipeline = safe_run_pipeline(st.session_state.cfg)
            st.session_state.pop("uploaded_graphs", None)
            st.rerun()

    with c2:
        if st.button("Clear uploaded graphs and rerun demo"):
            st.session_state.pop("uploaded_graphs", None)
            st.session_state.pipeline = safe_run_pipeline(st.session_state.cfg)
            st.rerun()

    st.stop()

tab_video, tab_align, tab_conflicts, tab_eval, tab_safety, tab_artifacts = st.tabs(
    [
        "Live demo video",
        "Entity alignment",
        "Conflicts and truth",
        "Evaluation",
        "Safety drill",
        "Artifacts and logs",
    ]
)

with tab_video:
    st.subheader("Automatically generated live demo")
    st.write(
        "This animated GIF is generated locally by the app on run. It shows source graphs, "
        "candidate alignment, conflict highlighting, and review/resolution states. "
        "No external video service is used."
    )

    st.image(pipeline["gif_bytes"], use_container_width=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Download live demo GIF",
            data=pipeline["gif_bytes"],
            file_name="kgdr_live_demo.gif",
            mime="image/gif",
        )

    with c2:
        st.download_button(
            "Download last frame PNG",
            data=pipeline["png_bytes"],
            file_name="kgdr_last_frame.png",
            mime="image/png",
        )

    with c3:
        st.metric("Config hash", pipeline["config_hash"][:16])

with tab_align:
    st.subheader("Graphs and candidate alignment")

    left, right = st.columns(2)

    with left:
        st.markdown(f"**Graph A: `{pipeline['graph_a'].source_id}`**")

        nodes_a_df = pd.DataFrame(
            [
                {
                    "node_id": n.node_id,
                    "label": n.label,
                    "category": n.category,
                    "aliases": ", ".join(n.aliases),
                    "properties": ", ".join(n.properties.keys()),
                    "provenance": n.provenance.source_document,
                    "confidence": n.provenance.confidence,
                }
                for n in pipeline["graph_a"].nodes
            ]
        )

        st.dataframe(nodes_a_df, use_container_width=True, hide_index=True)

    with right:
        st.markdown(f"**Graph B: `{pipeline['graph_b'].source_id}`**")

        nodes_b_df = pd.DataFrame(
            [
                {
                    "node_id": n.node_id,
                    "label": n.label,
                    "category": n.category,
                    "aliases": ", ".join(n.aliases),
                    "properties": ", ".join(n.properties.keys()),
                    "provenance": n.provenance.source_document,
                    "confidence": n.provenance.confidence,
                }
                for n in pipeline["graph_b"].nodes
            ]
        )

        st.dataframe(nodes_b_df, use_container_width=True, hide_index=True)

    st.markdown("**Match candidates**")

    matches_df = pd.DataFrame([m.model_dump(mode="json") for m in pipeline["matches"]])

    if not matches_df.empty:
        display_cols = [
            "left_id",
            "right_id",
            "left_label",
            "right_label",
            "score",
            "band",
            "id_score",
            "label_score",
            "alias_score",
            "embedding_score",
            "neighborhood_score",
        ]

        st.dataframe(matches_df[display_cols], use_container_width=True, hide_index=True)

        auto_df = matches_df[matches_df["band"] == "auto_accept"]
        review_df = matches_df[matches_df["band"] == "review"]

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Auto-accept candidates", len(auto_df))

        with c2:
            st.metric("Review candidates", len(review_df))

        with c3:
            st.metric("Total candidates", len(matches_df))
    else:
        st.warning("No match candidates were generated.")

with tab_conflicts:
    st.subheader("Conflict detection and ranked claims")

    conflicts = pipeline["conflicts"]

    if conflicts:
        conflict_rows = []
        claim_rows = []

        for c in conflicts:
            conflict_rows.append(
                {
                    "pair_id": c.pair_id,
                    "property": c.property_name,
                    "type": c.conflict_type,
                    "functional": c.functional,
                    "status": c.status,
                    "weighted_winner": c.weighted_winner or "",
                    "majority_winner": c.majority_winner or "",
                    "agreement": c.agreement,
                    "claims_retained": len(c.claims),
                    "rationale": c.rationale,
                }
            )

            for claim in c.claims:
                claim_rows.append(
                    {
                        "pair_id": c.pair_id,
                        "property": c.property_name,
                        "source_id": claim.source_id,
                        "node_id": claim.node_id,
                        "value": claim.value,
                        "observed_at": claim.observed_at.isoformat(),
                        "confidence": claim.confidence,
                        "reliability": claim.reliability,
                        "weight": claim.weight,
                        "source_document": claim.provenance.source_document,
                        "extractor_version": claim.provenance.extractor_version,
                    }
                )

        st.dataframe(pd.DataFrame(conflict_rows), use_container_width=True, hide_index=True)

        with st.expander("Ranked claims with provenance", expanded=False):
            st.dataframe(pd.DataFrame(claim_rows), use_container_width=True, hide_index=True)

        review_conflicts = [c for c in conflicts if c.status == "review"]

        st.subheader("Human review queue")

        if review_conflicts:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "pair_id": c.pair_id,
                            "property": c.property_name,
                            "conflict_type": c.conflict_type,
                            "rationale": c.rationale,
                        }
                        for c in review_conflicts
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No conflicts currently require human review under the active thresholds.")
    else:
        st.info("No conflicts detected under the current configuration.")

with tab_eval:
    st.subheader("Evaluation on synthetic demo data")

    st.warning(
        "These metrics are computed from clearly labelled synthetic demo data. "
        "They are not real-world performance claims."
    )

    metrics = pipeline["metrics"]
    align = metrics.get("alignment", {})
    conflict_metrics = metrics.get("conflicts", {})
    shapes = metrics.get("shapes", {})
    runtime = pipeline["runtime"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if isinstance(align, dict) and "auto_f1" in align:
            st.metric("Alignment F1 (auto)", f"{align['auto_f1']:.3f}")
        else:
            st.metric("Alignment F1", "TO VERIFY")

    with c2:
        if isinstance(align, dict) and "incorrectly_collapsed_count" in align:
            st.metric("Incorrectly collapsed", align["incorrectly_collapsed_count"])
        else:
            st.metric("Incorrectly collapsed", "TO VERIFY")

    with c3:
        st.metric(
            "Conflicts retained",
            "Yes" if conflict_metrics.get("all_conflicting_claims_retained") else "No",
        )

    with c4:
        st.metric("Schema conformance", f"{shapes.get('schema_conformance', 0.0):.3f}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Truth-discovery accuracy", f"{conflict_metrics.get('truth_discovery_accuracy', 0.0):.3f}")

    with c2:
        st.metric("Majority-vote accuracy", f"{conflict_metrics.get('majority_vote_accuracy', 0.0):.3f}")

    with c3:
        st.metric("Runtime", f"{runtime.get('elapsed_seconds', 0.0):.2f}s")

    with c4:
        st.metric("Peak memory", f"{runtime.get('peak_memory_mb', 0.0):.1f} MB")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Throughput", f"{runtime.get('throughput_nodes_per_second', 0.0):.1f} nodes/s")

    with c2:
        st.metric("Failure rate this run", f"{runtime.get('failure_rate_this_run', 0.0):.3f}")

    with c3:
        st.metric("Recovery time", runtime.get("recovery_time", "TO VERIFY"))

    st.subheader("Error analysis by property")

    error_analysis = metrics.get("error_analysis_by_property", {})

    if error_analysis:
        ea_rows = []

        for prop, counts in error_analysis.items():
            row = {"property": prop}
            row.update(counts)
            ea_rows.append(row)

        st.dataframe(pd.DataFrame(ea_rows).fillna(0), use_container_width=True, hide_index=True)
    else:
        st.info("No conflict error analysis available for this run.")

    st.subheader("Self-checks")
    st.dataframe(pd.DataFrame(pipeline["self_checks"]), use_container_width=True, hide_index=True)

    st.subheader("Metrics explicitly not claimed by this prototype")
    st.json(metrics.get("not_implemented_or_to_verify", {}))

with tab_safety:
    st.subheader("Controlled failure drill")
    st.write(
        "This section demonstrates graceful failure handling. The drill is intentionally separated "
        "from the main pipeline so the working demo remains available."
    )

    drill = st.session_state.get("failure_drill")

    if drill is None:
        st.info("No failure drill has been run yet. Use the sidebar button 'Run failure drill'.")
    elif "error" in drill:
        st.error(f"`{drill['error']['code']}`: {drill['error']['message']}")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Error payload**")
            st.json(drill["error"])

        with c2:
            st.markdown("**Drill logs**")
            st.json(drill.get("logs", []))

        st.success("Expected behaviour: the main demo pipeline remains unaffected.")
    else:
        st.warning("Failure drill completed without raising the expected synthetic error.")

with tab_artifacts:
    st.subheader("Artifacts, fingerprints, and structured logs")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Input fingerprints**")
        st.json(pipeline["result_json"]["input_fingerprints"])

    with c2:
        st.markdown("**Configuration hash**")
        st.code(pipeline["config_hash"], language="text")

    st.markdown("**Result JSON preview**")

    preview = {
        "app_version": pipeline["result_json"]["app_version"],
        "schema_version": pipeline["result_json"]["schema_version"],
        "config": pipeline["result_json"]["config"],
        "match_count": len(pipeline["result_json"]["matches"]),
        "conflict_count": len(pipeline["result_json"]["conflicts"]),
        "artifact_path": pipeline.get("artifact_path"),
    }

    st.json(preview)

    st.download_button(
        "Download full result JSON",
        data=json.dumps(pipeline["result_json"], indent=2, sort_keys=True).encode("utf-8"),
        file_name="kgdr_result.json",
        mime="application/json",
    )

    with st.expander("Structured logs", expanded=False):
        st.json(pipeline["logs"])

    st.markdown("---")
    st.markdown("#### Known gaps in this single-file prototype")
    st.markdown(
        "- Full repository layout, pytest suite, Ruff/mypy gates, Alembic migrations, FastAPI service, "
        "Neo4j/Oxigraph backends, pySHACL shapes, sentence-transformers, FAISS/hnswlib, and multi-language "
        "support are not implemented here.\n"
        "- The animated demo is a GIF generated locally to avoid FFmpeg dependencies.\n"
        "- Metrics are synthetic-demo metrics only. Real-world validation requires authorised data, gold labels, "
        "and clinical/domain governance where applicable."
    )
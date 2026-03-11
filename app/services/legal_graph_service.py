"""
Simple in-memory legal knowledge graph built from existing datasets.

Nodes:
  - acts
  - sections
  - judgments / cases
  - issues
  - templates

Edges:
  - cites
  - interprets
  - relates_to
  - applied_in
  - precedent_of
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from app.config import DATASETS_DIR


NodeType = Literal["act", "section", "judgment", "issue", "template"]
EdgeType = Literal["cites", "interprets", "relates_to", "applied_in", "precedent_of"]


@dataclass
class GraphNode:
    id: str
    type: NodeType
    data: Dict[str, Any]


@dataclass
class GraphEdge:
    source: str
    target: str
    type: EdgeType


class LegalGraphService:
    """
    Lightweight graph model built from JSONL datasets.
    Note: relationships depend on fields present in the datasets
    (e.g. 'sections_cited', 'cases_cited', 'issues', 'related_issues').
    """

    def __init__(
        self,
        acts_path: Optional[Path] = None,
        judgments_path: Optional[Path] = None,
        templates_path: Optional[Path] = None,
    ) -> None:
        self.acts_path = acts_path or (DATASETS_DIR / "bare_acts.jsonl")
        self.judgments_path = judgments_path or (DATASETS_DIR / "judgments.jsonl")
        self.templates_path = templates_path or (DATASETS_DIR / "draft_templates.jsonl")
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self._built = False

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    def _add_node(self, node_id: str, node_type: NodeType, data: Dict[str, Any]) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = GraphNode(id=node_id, type=node_type, data=data)

    def _add_edge(self, source: str, target: str, edge_type: EdgeType) -> None:
        self.edges.append(GraphEdge(source=source, target=target, type=edge_type))

    def _build_graph(self) -> None:
        if self._built:
            return
        # Acts and sections
        for rec in self._load_jsonl(self.acts_path):
            act_name = rec.get("act_name") or rec.get("act") or ""
            section_no = rec.get("section_number") or rec.get("section") or ""
            act_id = f"act:{act_name}"
            section_id = f"section:{act_name}:{section_no}"
            self._add_node(act_id, "act", {"name": act_name})
            self._add_node(section_id, "section", rec)
            self._add_edge(section_id, act_id, "relates_to")

        # Judgments / cases
        for rec in self._load_jsonl(self.judgments_path):
            case_name = rec.get("case_name") or rec.get("title") or ""
            citation = rec.get("citation") or ""
            judgment_id = f"judgment:{citation or case_name}"
            self._add_node(judgment_id, "judgment", rec)

            # Edges: interprets / applies sections if sections_cited present
            for sec in rec.get("sections_cited", []):
                # Expected form: {"act_name": "...", "section_number": "..."}
                act_name = sec.get("act_name") or sec.get("act") or ""
                section_no = sec.get("section_number") or sec.get("section") or ""
                section_id = f"section:{act_name}:{section_no}"
                if section_id in self.nodes:
                    self._add_edge(judgment_id, section_id, "interprets")

            # Edges: precedent_of / cites if cases_cited present
            for cited in rec.get("cases_cited", []):
                cited_id = f"judgment:{cited}"
                if cited_id in self.nodes:
                    self._add_edge(judgment_id, cited_id, "cites")

            # Issues as separate nodes, if present
            for issue in rec.get("issues_list", []):
                issue_id = f"issue:{issue}"
                self._add_node(issue_id, "issue", {"name": issue})
                self._add_edge(judgment_id, issue_id, "relates_to")

        # Templates
        for rec in self._load_jsonl(self.templates_path):
            tmpl_id = f"template:{rec.get('name') or rec.get('document_type')}"
            self._add_node(tmpl_id, "template", rec)
            for issue in rec.get("issues", []) or []:
                issue_id = f"issue:{issue}"
                self._add_node(issue_id, "issue", {"name": issue})
                self._add_edge(tmpl_id, issue_id, "relates_to")

        self._built = True

    # ---- Query helpers ----

    def find_cases_interpreting_section(self, act_name: str, section_no: str) -> List[Dict[str, Any]]:
        """
        Return judgments that have an 'interprets' edge to the given section.
        """
        self._build_graph()
        section_id = f"section:{act_name}:{section_no}"
        out: List[Dict[str, Any]] = []
        for e in self.edges:
            if e.type == "interprets" and e.target == section_id:
                node = self.nodes.get(e.source)
                if node:
                    out.append(node.data)
        return out

    def get_most_cited_cases(self, act_name: str, section_no: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Return most cited judgments for a given section based on in-graph 'cites' edges.
        """
        self._build_graph()
        section_id = f"section:{act_name}:{section_no}"
        # Find judgments interpreting the section
        interpreting_judgments = {
            e.source
            for e in self.edges
            if e.type == "interprets" and e.target == section_id
        }
        # Count how often those judgments are cited
        counts: Dict[str, int] = {jid: 0 for jid in interpreting_judgments}
        for e in self.edges:
            if e.type == "cites" and e.target in counts:
                counts[e.target] += 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result: List[Dict[str, Any]] = []
        for jid, c in ranked:
            node = self.nodes.get(jid)
            if node:
                data = dict(node.data)
                data["citation_count"] = c
                result.append(data)
        return result

    def get_cases_citing_judgment(self, case_or_citation: str) -> List[Dict[str, Any]]:
        """
        Return judgments that cite the given judgment.
        """
        self._build_graph()
        target_id = f"judgment:{case_or_citation}"
        out: List[Dict[str, Any]] = []
        for e in self.edges:
            if e.type == "cites" and e.target == target_id:
                node = self.nodes.get(e.source)
                if node:
                    out.append(node.data)
        return out

    def build_citation_network(self) -> List[Dict[str, Any]]:
        """
        Return a lightweight representation of the full citation network.
        Useful for analysis and tooling (not for direct prompting).
        """
        self._build_graph()
        network: List[Dict[str, Any]] = []
        for e in self.edges:
            if e.type == "cites":
                src = self.nodes.get(e.source)
                tgt = self.nodes.get(e.target)
                if src and tgt:
                    network.append(
                        {
                            "source": src.data.get("citation") or src.data.get("case_name"),
                            "target": tgt.data.get("citation") or tgt.data.get("case_name"),
                        }
                    )
        return network

    def find_sections_for_issue(self, issue: str) -> List[Dict[str, Any]]:
        """
        Sections related to a legal issue via issue nodes.
        """
        self._build_graph()
        issue_id = f"issue:{issue}"
        related_sections: List[Dict[str, Any]] = []
        for e in self.edges:
            # judgment/template -> issue, and judgment/template -> section
            if e.type == "relates_to" and e.target == issue_id:
                source = e.source
                for e2 in self.edges:
                    if e2.source == source and self.nodes.get(e2.target, None) and self.nodes[e2.target].type == "section":
                        related_sections.append(self.nodes[e2.target].data)
        return related_sections

    def find_precedents_cited_by_judgment(self, case_or_citation: str) -> List[Dict[str, Any]]:
        """
        Find judgments cited by a given judgment.
        """
        self._build_graph()
        judgment_id = f"judgment:{case_or_citation}"
        out: List[Dict[str, Any]] = []
        for e in self.edges:
            if e.type == "cites" and e.source == judgment_id:
                node = self.nodes.get(e.target)
                if node:
                    out.append(node.data)
        return out

    def build_context_snippets_for_query(self, query: str) -> str:
        """
        High-level graph context string for inclusion in prompts.
        Uses approximate matching against node data.
        """
        self._build_graph()
        q = query.lower()
        snippets: List[str] = []
        for node in self.nodes.values():
            text = json.dumps(node.data, ensure_ascii=False).lower()
            if q in text:
                if node.type == "section":
                    act = node.data.get("act_name") or node.data.get("act") or ""
                    num = node.data.get("section_number") or node.data.get("section") or ""
                    title = node.data.get("title", "")
                    snippets.append(f"Section node: {act} {num} - {title}")
                elif node.type == "judgment":
                    name = node.data.get("case_name") or node.data.get("title") or ""
                    citation = node.data.get("citation", "")
                    snippets.append(f"Judgment node: {name} ({citation})")
                elif node.type == "issue":
                    snippets.append(f"Issue node: {node.data.get('name')}")
        return "\n".join(snippets[:20])


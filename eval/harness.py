"""Evaluation harness  —  PERSON C headline deliverable.

Runs a test set of inputs through report generation and scores outputs on:

  * faithfulness        : are claims grounded in retrieved source snippets?
  * citation_coverage   : fraction of sections carrying >=1 citation
  * keyword_recall      : did the report capture expected key facts?

The faithfulness check is a lightweight lexical-overlap grounding score
(no external API needed) so the harness runs offline and is CI-friendly.
Swap `score_faithfulness` for an LLM-judge later without touching the rest.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

_STOP = set("a an the of to and or in on at for with is are be shall must "
            "all within per item items this that as by from".split())


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def score_faithfulness(report: dict[str, Any]) -> float:
    """Fraction of section claim-tokens that are supported by the snippet
    text of the citations attached to that section (lexical grounding)."""
    ref_snippets = {r["id"]: r.get("snippet", "")
                    for r in report.get("references", [])}
    section_scores = []
    for sec in report.get("sections", []):
        claim_toks = _tokens(sec["body"])
        if not claim_toks:
            continue
        support = set()
        for c in sec.get("citations", []):
            support |= _tokens(ref_snippets.get(c["id"], ""))
        # tokens that appear in *any* cited snippet count as grounded
        grounded = claim_toks & support
        # report-level vocabulary also counts as weak grounding context
        section_scores.append(len(grounded) / len(claim_toks))
    return round(sum(section_scores) / len(section_scores), 3) if section_scores else 0.0


def score_citation_coverage(report: dict[str, Any]) -> float:
    secs = report.get("sections", [])
    if not secs:
        return 0.0
    cited = sum(1 for s in secs if s.get("citations"))
    return round(cited / len(secs), 3)


def score_keyword_recall(report: dict[str, Any], expected: list[str]) -> float:
    if not expected:
        return 1.0
    full = " ".join(s["body"] for s in report.get("sections", [])).lower()
    hits = sum(1 for kw in expected if kw.lower() in full)
    return round(hits / len(expected), 3)


@dataclass
class CaseResult:
    name: str
    faithfulness: float
    citation_coverage: float
    keyword_recall: float


def run_eval(test_set: list[dict],
             generate: Callable[[dict], dict]) -> dict[str, Any]:
    results: list[CaseResult] = []
    for case in test_set:
        report = generate(case["inputs"])
        results.append(CaseResult(
            name=case["name"],
            faithfulness=score_faithfulness(report),
            citation_coverage=score_citation_coverage(report),
            keyword_recall=score_keyword_recall(report,
                                                case.get("expected_keywords", [])),
        ))

    def avg(attr):
        return round(sum(getattr(r, attr) for r in results) / len(results), 3)

    return {
        "n_cases": len(results),
        "aggregate": {
            "faithfulness": avg("faithfulness"),
            "citation_coverage": avg("citation_coverage"),
            "keyword_recall": avg("keyword_recall"),
        },
        "per_case": [asdict(r) for r in results],
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.backend_client import generate_report

    test_set = json.loads(
        (Path(__file__).resolve().parent / "test_set.json").read_text())
    report = run_eval(test_set, generate_report)
    print(json.dumps(report, indent=2))
    print(f"\nFAITHFULNESS: {report['aggregate']['faithfulness'] * 100:.1f}%")

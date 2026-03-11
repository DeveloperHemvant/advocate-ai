"""
Generic judgment crawler for Indian court websites.

NOTE: This is a template implementation; individual court websites may
require custom parsing logic. Networking must be configured so that the
container / host running this script can reach the public sites.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup  # type: ignore[import]

from app.config import DATASETS_DIR
from app.scripts.vector_index_management import rebuild_full_index


@dataclass
class CrawledJudgment:
    case_name: str
    court: str
    year: Optional[int]
    citation: str
    text: str


class JudgmentCrawler:
    """
    Minimal crawler that can be extended with site-specific logic.
    """

    def __init__(self, dataset_path: Optional[Path] = None) -> None:
        self.dataset_path = dataset_path or (DATASETS_DIR / "judgments.jsonl")

    def fetch_html(self, url: str) -> str:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.text

    def parse_judgment_from_html(self, html: str) -> CrawledJudgment:
        """
        Very generic HTML parser: expects case name in <title> and text in <body>.
        For real deployments, replace with specific logic per court site.
        """
        soup = BeautifulSoup(html, "html.parser")
        title = (soup.title.string or "").strip() if soup.title else "Unknown v. Unknown"
        body_text = soup.get_text(separator="\n")
        return CrawledJudgment(
            case_name=title,
            court="Unknown Court",
            year=None,
            citation="",
            text=body_text,
        )

    def save_judgment(self, judgment: CrawledJudgment) -> None:
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(judgment), ensure_ascii=False) + "\n")

    def crawl_and_store(self, url: str) -> None:
        html = self.fetch_html(url)
        judgment = self.parse_judgment_from_html(html)
        self.save_judgment(judgment)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Crawl a judgment URL and update the Legal AI datasets.")
    parser.add_argument("url", type=str, help="URL of the judgment page.")
    args = parser.parse_args()

    crawler = JudgmentCrawler()
    crawler.crawl_and_store(args.url)

    print("Crawled and stored judgment. Rebuilding unified vector index...")
    count = rebuild_full_index()
    print(f"Rebuilt vector index with {count} items.")


if __name__ == "__main__":
    main()


from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class TimelineEvent:
    label: str
    date: Optional[str]
    description: str


DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
]


class TimelineService:
    """
    Extracts simple chronological events from case text using regex + heuristics.
    """

    def extract_events(self, case_text: str) -> List[TimelineEvent]:
        events: List[TimelineEvent] = []
        lines = [l.strip() for l in case_text.splitlines() if l.strip()]
        for line in lines:
            lower = line.lower()
            label = None
            if "contract" in lower and "sign" in lower:
                label = "contract_signed"
            elif "payment" in lower and ("due" in lower or "payable" in lower):
                label = "payment_due"
            elif "notice" in lower and ("sent" in lower or "issued" in lower or "served" in lower):
                label = "notice_sent"
            elif "case" in lower and ("filed" in lower or "instituted" in lower):
                label = "case_filed"
            if not label:
                continue
            date = None
            for pattern in DATE_PATTERNS:
                m = re.search(pattern, line)
                if m:
                    date = m.group(0)
                    break
            events.append(TimelineEvent(label=label, date=date, description=line))
        # Sort by parsed date where possible
        def sort_key(ev: TimelineEvent):
            if not ev.date:
                return datetime.max
            for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y", "%d/%m/%y", "%d-%m-%y"):
                try:
                    return datetime.strptime(ev.date, fmt)
                except ValueError:
                    continue
            return datetime.max

        events.sort(key=sort_key)
        return events


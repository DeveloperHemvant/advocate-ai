"""
CLI entry point to rebuild the unified FAISS vector index.
Indexes drafts, bare act sections, judgments, and templates.
"""

import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.scripts.vector_index_management import rebuild_full_index  # noqa: E402


if __name__ == "__main__":
    count = rebuild_full_index()
    print(f"Rebuilt vector index with {count} items.")


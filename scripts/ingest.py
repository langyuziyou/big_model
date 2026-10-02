# -*- coding: utf-8 -*-
"""命令行入库：python scripts/ingest.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.rag as rag  # noqa: E402


if __name__ == "__main__":
    result = rag.ingest(force=True)
    print("入库完成：")
    print("  集合:", result["collection"])
    print("  分块总数:", result["chunks"])
    for d in result["detail"]:
        print(f"  - {d['file']}: {d['chunks']} 块" + (f" ({d.get('warning')})" if d.get("warning") else ""))

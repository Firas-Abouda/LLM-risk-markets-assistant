from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import pandas as pd
import re
import sys

from utils_text import chunk_text

PROCESSED = Path("data/processed")
MANIFEST  = PROCESSED / "manifest.csv"
OUT       = PROCESSED / "corpus.parquet"

PARA_SPLIT = re.compile(r"\n{2,}|\.{6,}")  # blank-line blocks or long dotted leaders


def _read_manifest(path: Path) -> pd.DataFrame:
    """
    Load manifest and validate required columns.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"[build_corpus] ERROR: manifest not found at {path}", file=sys.stderr)
        raise

    required = {
        "ticker", "source", "period_label", "quarter", "section",
        "filename", "processed_path"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[build_corpus] manifest is missing columns: {sorted(missing)}")

    return df


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into coarse paragraphs. If the text is essentially one big block
    , return it as a single-element list.
    """
    paras = [seg.strip() for seg in PARA_SPLIT.split(text) if seg.strip()]
    return paras if len(paras) > 2 else [text.strip()]


def _build_rows(mf: pd.DataFrame) -> List[Dict]:
    """
    Iterate manifest rows, read cleaned files, split into paragraphs, then chunk.
    Produce one row per chunk with provenance metadata.
    """
    rows: List[Dict] = []

    for _, r in mf.iterrows():
        p = Path(str(r["processed_path"]))
        if not p.exists():
            # Soft warn; skip missing file
            print(f"[build_corpus] WARN: missing file -> {p}", file=sys.stderr)
            continue

        text = p.read_text(encoding="utf-8", errors="ignore")
        paragraphs = _split_into_paragraphs(text)

        para_id = 0
        for para in paragraphs:
            chunks = chunk_text(para, max_tokens=180, overlap=30)
            for cid, ch in enumerate(chunks):
                rows.append({
                    "ticker":        r.get("ticker", ""),
                    "doc_type":      r.get("source", ""),          # 10K / EarningsCall
                    "period_label": r.get("period_label", ""),     # FY2025 / CY2025
                    "quarter":       r.get("quarter", ""),
                    "section":       r.get("section", ""),         # RiskFactors / "" for calls
                    "source_file":   r.get("filename", ""),
                    "processed_path": r.get("processed_path", ""),
                    # local location
                    "para_id": para_id,
                    "chunk_id": cid,
                    # content
                    "text": ch,
                })
            para_id += 1

    return rows


def main() -> None:
    mf = _read_manifest(MANIFEST)
    rows = _build_rows(mf)

    if not rows:
        print("[build_corpus] No chunks produced. Check manifest and processed files.", file=sys.stderr)
        return

    df = pd.DataFrame(rows)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"[build_corpus] Saved {len(df):,} chunks -> {OUT}")

if __name__ == "__main__":
    main()
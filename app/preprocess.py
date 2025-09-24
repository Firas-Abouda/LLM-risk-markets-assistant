import os
import re
import csv
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, List

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"

RE_SPACE = re.compile(r"[ \t]+")
RE_BLANKLINES = re.compile(r"\n{3,}")
RE_DOTS = re.compile(r"\.{5,}|…")  # dot leaders + unicode ellipsis

# page markers
RE_PAGE_NUM = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
RE_PAGE_MARK = re.compile(r"^\s*[-–—]?\s*\d+\s*[-–—]?\s*$", re.MULTILINE)
RE_PAGE_WORD = re.compile(r"^\s*Page\s+\d+(\s+of\s+\d+)?\s*$",
                          re.IGNORECASE | re.MULTILINE)

# Simple filename parser: TICKER_DOC_YEAR_TYPE_...(.txt ) e.g., MSFT_10K_FY2025_RiskFactors.txt
def parse_filename(fname: str) -> Dict[str, str]:
    # Validate extension
    if not fname.lower().endswith(".txt"):
        raise ValueError(f"Unexpected file extension: {fname}")

    name = fname[:-4]  # strip ".txt"
    parts: List[str] = name.split("_")

    meta = {
        "ticker": parts[0] if len(parts) > 0 else "",
        "source": "",          # 10K or EarningsCall
        "period_label": "",    # FY2025 / CY2025
        "quarter": "",         # Q1..Q4 (for earnings)
        "section": "",         # RiskFactors, etc.
        "filename": fname,
    }

    leftovers = []
    for p in parts[1:]:  # skip ticker
        if p == "10K":
            meta["source"] = "10K"
        elif p == "EarningsCall":
            meta["source"] = "EarningsCall"
        elif p.startswith(("FY", "CY")):
            meta["period_label"] = p
        elif re.fullmatch(r"Q[1-4]", p):
            meta["quarter"] = p
        else:
            leftovers.append(p)

    # join leftovers into one "section" string if there are any
    if leftovers:
        meta["section"] = "_".join(leftovers)

    return meta

def _normalize_unicode(s: str) -> str:
    """
    Systemic normalization with some special-casing:
      1) NFKC fold.
      2) Convert all Unicode separators (Z*) to ASCII space, preserving '\n' and '\t'.
      3) Remove control characters (C*) except '\n' and '\t'.
      4) Map punctuation by *category*, with limited name checks:
         - Pd (any dash) and MINUS SIGN → '-'
         - Pi/Pf (opening/closing quotes): DOUBLE → '"', else → "'"
         - Po with name containing 'BULLET' or 'MIDDLE DOT' → '-'
         - Po with name containing 'ELLIPSIS' → '...'
    """
    # normalize
    s = s.replace("\r", "\n")
    s = unicodedata.normalize("NFKC", s)

    out_chars = []
    for ch in s:
        if ch in ("\n", "\t"):
            out_chars.append(ch)
            continue

        cat = unicodedata.category(ch)

        # all Unicode separators -> plain space
        if cat.startswith("Z"):  # Zs, Zl, Zp
            out_chars.append(" ")
            continue

        # drop controls (Cc, Cf, Cs, Co, Cn)
        if cat.startswith("C"):
            continue

        # any dash punctuation
        if cat == "Pd":  # any dash punctuation
            out_chars.append("-")
            continue

        if cat in ("Pi", "Pf"):  # opening/closing quotes
            # Use name to decide single vs double
            name = unicodedata.name(ch, "")
            out_chars.append('"' if "DOUBLE" in name else "'")
            continue

        if cat.startswith("P"):  # general punctuation

            name = unicodedata.name(ch, "")

            if "ELLIPSIS" in name:
                out_chars.append("...")
                continue

            if "BULLET" in name or "MIDDLE DOT" in name:
                out_chars.append("-")
                continue

            if name == "MINUS SIGN":  # U+2212
                out_chars.append("-")
                continue

            # else: keep punctuation as-is (.,:;()/%$ etc.)
            out_chars.append(ch)
            continue


        out_chars.append(ch)

    return "".join(out_chars)

def basic_clean(text: str, debug: bool = False) -> str:

    text = _normalize_unicode(text)

    text = RE_DOTS.sub("...", text)
    text = RE_SPACE.sub(" ", text)

    # track removals if debug mode
    if debug:
        removed = []
        for regex, label in [
            (RE_PAGE_NUM, "PAGE_NUM"),
            (RE_PAGE_MARK, "PAGE_MARK"),
            (RE_PAGE_WORD, "PAGE_WORD"),
        ]:
            for m in regex.findall(text):
                removed.append((label, repr(m)))
        if removed:
            print("⚠️ Removed lines:")
            for label, val in removed:
                print(f"  {label}: {val}")

    # remove page artifacts
    text = RE_PAGE_NUM.sub("", text)
    text = RE_PAGE_MARK.sub("", text)
    text = RE_PAGE_WORD.sub("", text)

    # collapse blank lines
    text = RE_BLANKLINES.sub("\n\n", text)

    return text.strip()

def load_and_clean(debug: bool = False) -> List[Dict[str, str]]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, str]] = []

    for fname in sorted(os.listdir(RAW_DIR), key=str.lower):
        if not fname.lower().endswith(".txt"):
            continue

        raw_path = RAW_DIR / fname
        try:
            with open(raw_path, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()

            cleaned = basic_clean(raw_text, debug=debug)

            out_path = os.path.join(PROCESSED_DIR, fname)
            with open(out_path, "w", encoding="utf-8", newline="") as out:
                out.write(cleaned)

            meta = parse_filename(fname)
            meta.update({
                "chars": len(cleaned),
                "processed_path": out_path,
                "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            })
            records.append(meta)

        except Exception as e:
            records.append({
                "ticker": "",
                "source": "",
                "period_label": "",
                "quarter": "",
                "section": "",
                "filename": fname,
                "chars": 0,
                "processed_path": "",
                "ingested_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "error": f"{type(e).__name__}: {e}",
            })

    if records:
        has_error = any("error" in r for r in records)
        fieldnames = ["ticker","source","period_label","quarter","section",
                      "filename","chars","processed_path","ingested_at"] + (["error"] if has_error else [])
        with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                for k in fieldnames:
                    r.setdefault(k, "")
                writer.writerow(r)

    return records

if __name__ == "__main__":
    recs = load_and_clean(debug=True)
    print(f"Processed {len(recs)} documents into {PROCESSED_DIR}")
    if recs:
        print("Example manifest row:", recs[0])
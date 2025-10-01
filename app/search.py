import argparse, numpy as np, joblib, pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import linear_kernel

ART = Path("artifacts") / "tfidf_index.joblib"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--company", choices=["MSFT","NVDA","JPM"], default=None)
    ap.add_argument("--doc", choices=["10K","EarningsCall"], default=None)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    b = joblib.load(ART)
    pipe, X, df = b["pipe"], b["X"], b["meta"]

    #filter based on args
    mask = pd.Series(True, index=df.index)
    if args.company: mask &= (df["ticker"]==args.company)
    if args.doc:     mask &= (df["doc_type"]==args.doc)

    idx = np.where(mask.values)[0]
    if len(idx)==0: print("No docs match filters."); return

    #query vectorization
    qv = pipe.transform([args.q])

    #dot product with normalized vector -> cosine similarity
    sims = linear_kernel(qv, X[idx]).ravel()

    top = sims.argsort()[::-1][:args.top_k]

    for r, j in enumerate(top, 1):
        row = df.iloc[idx[j]]
        print(f"\n[{r}] score={sims[j]:.3f} | {row['ticker']} | {row['doc_type']} | {row['period_label']} {row.get('quarter','')}")
        print(f"src={row['source_file']}  para={row['para_id']}  chunk={row['chunk_id']}")
        print("-"*80)
        print(row["text"])

if __name__ == "__main__":
    main()
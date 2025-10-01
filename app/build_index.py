from pathlib import Path
import pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

CORPUS = Path("data/processed/corpus.parquet")
ART    = Path("artifacts"); ART.mkdir(exist_ok=True)

def main() -> None:
    df = pd.read_parquet(CORPUS)

   # Add context from metadata to help matching
    df["text_for_index"] = (
        df["text"].astype(str) + " " +
        df["ticker"].astype(str) + " " +
        df["doc_type"].astype(str) + " " +
        df["period_label"].astype(str) + " " +
        df["section"].astype(str)
    )


    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),     # unigrams + bigrams
            max_df=0.9,             # ignore terms that appear in >90% of chunks (too common)
            min_df=2,               # ignore terms seen in only 1 chunk (noise)
            sublinear_tf=True       # log-scale TF to dampen very frequent words
        ))
    ])

    X = pipe.fit_transform(df["text_for_index"])

    joblib.dump({"pipe": pipe, "X": X, "meta": df}, ART / "tfidf_index.joblib")
    print(f"Indexed {X.shape[0]} chunks; vocab size={X.shape[1]} -> {ART/'tfidf_index.joblib'}")

if __name__ == "__main__":
    main()
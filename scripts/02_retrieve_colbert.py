"""Step 2: retrieve semantically similar seeker posts with ColBERT (via RAGatouille).

For every evaluation instance, the seeker post is used as the query against an index of
learning-set seeker posts (max. 256 tokens). The top 20 candidates are stored; smaller
retrieval depths (k = 5, 10) are taken as prefixes of this ranking in step 3.

Output:
    data/retrieval/colbert_top20.csv  columns: test_id_query, similar_id_query, similarity
"""

import argparse

import pandas as pd
from ragatouille import RAGPretrainedModel

from common import PROCESSED_DIR, RETRIEVAL_DIR, ensure_dirs

INDEX_NAME = "seeker_posts_index"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--checkpoint", default="colbert-ir/colbertv2.0")
    args = parser.parse_args()

    learning = pd.read_csv(PROCESSED_DIR / "learning.csv")
    evaluation = pd.read_csv(PROCESSED_DIR / "evaluation.csv")

    rag = RAGPretrainedModel.from_pretrained(args.checkpoint)
    rag.index(
        collection=learning["seeker_post"].astype(str).tolist(),
        document_ids=learning["id_query"].tolist(),
        index_name=INDEX_NAME,
        max_document_length=256,
        split_documents=False,
    )

    rows = []
    for _, row in evaluation.iterrows():
        for hit in rag.search(query=str(row["seeker_post"]), index_name=INDEX_NAME, k=args.top_k):
            rows.append({
                "test_id_query": row["id_query"],
                "similar_id_query": hit["document_id"],
                "similarity": hit["score"],
            })

    ensure_dirs(RETRIEVAL_DIR)
    out = RETRIEVAL_DIR / f"colbert_top{args.top_k}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {len(rows)} retrieval results to {out}")


if __name__ == "__main__":
    main()

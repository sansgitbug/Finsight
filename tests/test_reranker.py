from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker

QUERY = "How did iPhone revenue change?"

retriever = Retriever()
results = retriever.retrieve_from_text(QUERY)

print(f"\nRetrieved {len(results)} candidates.\n")

reranker = Reranker()
reranked = reranker.rerank(QUERY, results)

print("=" * 100)

for candidate in reranked:
    print(f"Old Rank        : {candidate.rank_before}")
    print(f"New Rank        : {candidate.rank_after}")
    print(f"Retrieval Score : {candidate.retrieval_score:.4f}")
    print(f"Reranker Score  : {candidate.reranker_score:.4f}")

    chunk = candidate.result.chunk

    print(f"Section         : {chunk.section_name}")
    print(f"Filing Date     : {chunk.filing_date}")

    print(chunk.text[:300])
    print("-" * 100)
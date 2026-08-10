from src.preprocessing.chunker import Chunk
from src.retrieval.bm25 import BM25Retriever


def test_bm25_returns_results():

    chunks = [
        Chunk(
            chunk_id="1",
            ticker="AAPL",
            filing_date="2025",
            filing_type="10-K",
            section_name="Revenue",
            text="Apple revenue increased significantly.",
            token_count=4,
            start_line=1,
            end_line=1,
            character_start=0,
            character_end=37,
            metadata={},
        ),
        Chunk(
            chunk_id="2",
            ticker="AAPL",
            filing_date="2025",
            filing_type="10-K",
            section_name="Risk",
            text="Supply chain risks remain.",
            token_count=4,
            start_line=2,
            end_line=2,
            character_start=38,
            character_end=65,
            metadata={},
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "apple revenue",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "1"
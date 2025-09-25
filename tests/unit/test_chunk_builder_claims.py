from patent_rag_app.ingestion.chunk_builder import build_patent_chunks
from patent_rag_app.ingestion.schemas import ClaimRecord, PatentFrontMatter


def test_build_claim_chunks_includes_claim_text() -> None:
    document = PatentFrontMatter(
        patent_id="TEST123",
        total_pages=5,
        front_page_fields={},
        source_path="/tmp/test.pdf",
        claims=[
            ClaimRecord(
                claim_id="claim_0001",
                number="1",
                text="A sample claim text",
                page_start=10,
                page_end=11,
            )
        ],
    )

    chunks = build_patent_chunks(document)
    claim_chunks = [chunk for chunk in chunks if chunk.content_type == "claim"]

    assert len(claim_chunks) == 1
    claim_chunk = claim_chunks[0]
    assert claim_chunk.section_code == "claim_0001"
    assert claim_chunk.section_label == "Claim 1"
    assert claim_chunk.text == "A sample claim text"
    assert claim_chunk.page_start == 10
    assert claim_chunk.page_end == 11

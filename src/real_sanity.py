"""Bounded real OHRBench QA → embedding → FAISS → page relevance trace."""

from collections import defaultdict

from chunkers import load_tokenizer
from embeddings import embed_chunk_texts, embed_query, load_embedding_model
from faiss_index import add_vectors, create_index
from official_qa import load_mapped_qas
from ohrbench_adapter import iter_documents, load_pages
from retrieval_metrics import is_page_relevant, score_ranking
from retrieval_pipeline import chunk_document, retrieve_one


def main():
    pages = load_pages("clean")
    qas, _ = load_mapped_qas(pages[["doc_name", "page_idx", "gt_text"]])
    by_doc = defaultdict(list)
    for qa in qas:
        by_doc[qa["doc_name"]].append(qa)
    multi_doc = "administration/DUDE_54c771a3fc5da43da9e53e6854ef8a77"
    multi = sorted(by_doc[multi_doc], key=lambda q: q["query_id"])[0]
    page_counts = pages.groupby("doc_name")["gt_text"].agg(lambda texts: sum(map(len, texts)))
    single = min((q for q in qas if len(q["evidence_pages"]) == 1
                  and q["doc_name"] != multi_doc and page_counts[q["doc_name"]] < 3000),
                 key=lambda q: q["query_id"])
    distractor = next(doc for doc in sorted(page_counts.index)
                      if doc not in {multi_doc, single["doc_name"]} and page_counts[doc] < 3000)
    wanted = {multi_doc, single["doc_name"], distractor}
    model = load_embedding_model()
    tokenizer = load_tokenizer()
    index = create_index()
    metadata = []
    for document in iter_documents(pages[pages["doc_name"].isin(wanted)], "clean"):
        chunks = chunk_document(document, "fixed", tokenizer, model)
        metadata.extend(chunks)
        for start in range(0, len(chunks), 32):
            add_vectors(index, embed_chunk_texts([c["text"] for c in chunks[start:start + 32]], model))
    assert index.ntotal == len(metadata) and index.ntotal > 0
    for qa in (multi, single):
        assert any(is_page_relevant(chunk, qa) for chunk in metadata)
        ranking = retrieve_one(index, metadata, embed_query(qa["question"], model), k=min(100, index.ntotal))
        metrics = score_ranking([chunk for _, chunk in ranking], qa)
        assert len(ranking) == index.ntotal
        assert all(chunk["source_page_ids"] for _, chunk in ranking)
        print({"query_id": qa["query_id"], "question": qa["question"],
               "doc_name": qa["doc_name"], "evidence_pages": qa["evidence_pages"],
               "top_chunk_document": ranking[0][1]["document_id"],
               "top_chunk_pages": ranking[0][1]["source_page_ids"],
               "top_score": ranking[0][0], **metrics})
    print({"documents": len(wanted), "indexed_chunks": index.ntotal, "status": "sanity_passed"})


if __name__ == "__main__":
    main()

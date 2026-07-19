from rag_pipeline import retriever, llm
from datetime import datetime

test_queries = ["what error in GCP", "gcp"]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"QUERY: '{query}'")
    print(f"{'='*60}")
    
    docs = retriever.invoke(query)
    print(f"Retrieved {len(docs)} chunks:\n")
    for i, doc in enumerate(docs, 1):
        print(f"[Chunk {i}] Source: {doc.metadata.get('file','?')}")
        print(doc.page_content[:300])
        print("-"*40)

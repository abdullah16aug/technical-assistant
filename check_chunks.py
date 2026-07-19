import csv
import rag_pipeline

dataset_path = "data/eval/rag_test_dataset.csv"

with open(dataset_path, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    first_row = next(reader)

question = first_row['question']
expected_chunk = first_row['relevant_chunk']

print("=========================================")
print(f"QUESTION: {question}")
print("=========================================\n")

print("--- EXPECTED RELEVANT CHUNK (From Dataset) ---")
print(expected_chunk)
print("\n" + "="*41 + "\n")

retrieved_docs = rag_pipeline.retriever.invoke(question)

print("--- ACTUAL RETRIEVED CHUNKS (From RAG Pipeline) ---")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n[Chunk {i}] (Source: {doc.metadata.get('file', 'Unknown')})")
    print(doc.page_content)
    print("-" * 20)

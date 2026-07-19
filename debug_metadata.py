import csv
from rag_pipeline import vectorstore

def main():
    print("--- 1. Unique 'file' metadata in Chroma DB ---")
    all_docs = vectorstore._collection.get(include=["metadatas"])
    metadatas = all_docs.get("metadatas", [])
    
    unique_files = set()
    for m in metadatas:
        if m and "file" in m:
            unique_files.add(m["file"])
            
    for f in unique_files:
        print(f"DB Source: {f}")
        
    print("\n--- 2. Unique 'source_file' in Dataset ---")
    dataset_path = "data/eval/rag_test_dataset.csv"
    unique_ds_files = set()
    with open(dataset_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_ds_files.add(row["source_file"])
            
    for f in unique_ds_files:
        print(f"Dataset Source: {f}")

if __name__ == "__main__":
    main()

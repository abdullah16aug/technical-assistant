import sys
# Import the initialized vectorstore from your engine file
from rag_pipeline import vectorstore

def inspect_vector_database():
    # 🗄️ 1. Get the total number of items in the collection
    total_chunks = vectorstore._collection.count()
    print("=" * 50)
    print(f"📊 CURRENT DATABASE STATUS")
    print(f"Total text chunks stored: {total_chunks}")
    print("=" * 50)
    
    if total_chunks == 0:
        print("⚠️ The database is empty. Please run your ingestion endpoints first!")
        return

    # ⌨️ 2. Accept a random string input from the terminal
    query = input("\nEnter a search phrase to test under-the-hood matching: ")
    if not query.strip():
        print("Empty query. Exiting.")
        return

    print(f"\n🔍 Searching for matches to: '{query}'...\n")
    
    # 🌌 3. Perform a similarity search that returns both documents and scores
    results = vectorstore.similarity_search_with_score(query, k=3)
    
    # 📋 4. Print out the text and the raw mathematical distance
    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- Match #{i} (Distance Score: {score:.4f}) ---")
        print(f"Source Type: {doc.metadata.get('source_type', 'unknown')}")
        print(f"File: {doc.metadata.get('file', 'unknown')}")
        print(f"Content Preview:\n{doc.page_content}")
        print("-" * 50)

if __name__ == "__main__":
    inspect_vector_database()
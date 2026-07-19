import csv
import uuid
import time
import re
import rag_pipeline

def extract_score(text: str) -> float:
    """Extracts the first floating point number or integer from text."""
    match = re.search(r'\d+(\.\d+)?', text)
    if match:
        return float(match.group())
    return 0.0

def evaluate_faithfulness(context: str, answer: str) -> float:
    prompt = f"""You are an expert evaluator. 
Given the retrieved context and the generated answer, determine if the generated answer is strictly based on the provided context.
Respond with ONLY a single number: 1 if it is completely faithful to the context, 0 if it contains hallucinations or ungrounded information.

Context:
{context}

Generated Answer:
{answer}

Score:"""
    response = rag_pipeline.llm.invoke(prompt)
    return extract_score(response.content)

def evaluate_relevancy(question: str, reference_answer: str, answer: str) -> float:
    prompt = f"""You are an expert evaluator.
Given a question, a reference ground-truth answer, and a generated answer, determine how relevant and accurate the generated answer is compared to the reference answer.
Respond with ONLY a single number from 0.0 to 1.0, where 1.0 means fully relevant and accurate, and 0.0 means completely irrelevant or contradictory.

Question:
{question}

Reference Answer:
{reference_answer}

Generated Answer:
{answer}

Score:"""
    response = rag_pipeline.llm.invoke(prompt)
    return extract_score(response.content)


def evaluate_chunk_hit(true_chunk: str, retrieved_chunks: list) -> tuple[int, int]:
    """
    Uses LLM as a judge to check which of the retrieved chunks semantically
    contain the information from the ground-truth relevant_chunk.
    Returns (hit, relevant_retrieved_count).
    """
    chunks_text = ""
    for i, doc in enumerate(retrieved_chunks, 1):
        chunks_text += f"\n--- Chunk {i} ---\n{doc.page_content.strip()}\n"

    prompt = f"""You are an expert RAG evaluation judge.

Your task: Check each retrieved chunk below and determine if it contains the same core information as the Reference Chunk.

Reference Chunk (ground truth):
{true_chunk}

Retrieved Chunks:
{chunks_text}

For each chunk, respond with ONLY:
Chunk 1: YES or NO
Chunk 2: YES or NO
... and so on.

A chunk is YES if it contains the same key facts, steps, commands, or answers as the Reference Chunk — even if worded differently.
A chunk is NO if it is on a different topic or missing the key information."""

    response = rag_pipeline.llm.invoke(prompt)
    result_text = response.content.strip()

    # Parse YES/NO answers
    relevant_count = 0
    hit = 0
    lines = result_text.splitlines()
    for line in lines:
        if "YES" in line.upper():
            relevant_count += 1
            hit = 1

    return hit, relevant_count


def main():
    dataset_path = "data/eval/rag_test_dataset.csv"
    output_path = "data/eval/rag_eval_results.csv"
    
    print(f"Loading dataset from {dataset_path}...")
    
    results = []
    
    with open(dataset_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + [
            "llm_assistant_response",
            "retrieved_chunks",
            "hit_ratio",
            "precision",
            "recall",
            "faithfulness",
            "relevancy"
        ]
        
        rows = list(reader)
    
    total_hit = 0
    total_precision = 0.0
    total_recall = 0.0
    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_processed = 0
    
    print(f"Total records to process: {len(rows)}")
    
    for row in rows:
        question = row['question']
        reference_answer = row['reference_answer']
        true_chunk = row['relevant_chunk']
        
        session_id = str(uuid.uuid4())
        
        try:
            print(f"Processing ID: {row['id']}")
            # 1. Retrieval
            retrieved_docs = rag_pipeline.retriever.invoke(question)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # 2. Generation
            llm_assistant_response = rag_pipeline.generate_chat_response(session_id, question)

            # 3. Retrieval Metrics — LLM judges if relevant_chunk info is in any retrieved chunk
            retrieved_chunk_previews = []
            for doc in retrieved_docs:
                preview = doc.page_content.strip().replace("\n", " ")[:200]
                retrieved_chunk_previews.append(f"[{doc.metadata.get('file', 'unknown')}]: {preview}")

            hit, relevant_retrieved = evaluate_chunk_hit(true_chunk, retrieved_docs)
            precision = relevant_retrieved / len(retrieved_docs) if len(retrieved_docs) > 0 else 0.0
            recall = 1.0 if hit else 0.0

            # 4. LLM Judge Metrics
            faithfulness = evaluate_faithfulness(context_text, llm_assistant_response)
            relevancy = evaluate_relevancy(question, reference_answer, llm_assistant_response)

            # Save results
            row['llm_assistant_response'] = llm_assistant_response
            row['retrieved_chunks'] = " || ".join(retrieved_chunk_previews)
            row['hit_ratio'] = hit
            row['precision'] = precision
            row['recall'] = recall
            row['faithfulness'] = faithfulness
            row['relevancy'] = relevancy
            
            results.append(row)
            
            total_hit += hit
            total_precision += precision
            total_recall += recall
            total_faithfulness += faithfulness
            total_relevancy += relevancy
            total_processed += 1
            
            # Sleep slightly to avoid API throttling
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing ID {row['id']}: {e}")
            row['llm_assistant_response'] = f"ERROR: {str(e)}"
            row['hit_ratio'] = 0
            row['precision'] = 0.0
            row['recall'] = 0.0
            row['faithfulness'] = 0.0
            row['relevancy'] = 0.0
            results.append(row)

    # Write results to new CSV
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
            
    print(f"\nEvaluation complete. Results saved to {output_path}")
    
    if total_processed > 0:
        # Print summary
        print("\n--- Evaluation Summary ---")
        print(f"Average Hit Ratio: {total_hit / total_processed:.4f}")
        print(f"Average Precision: {total_precision / total_processed:.4f}")
        print(f"Average Recall: {total_recall / total_processed:.4f}")
        print(f"Average Faithfulness: {total_faithfulness / total_processed:.4f}")
        print(f"Average Relevancy: {total_relevancy / total_processed:.4f}")

if __name__ == "__main__":
    main()

import argparse
import json
import os
import sys

from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

# Add the current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rag_client
import llm_client
import ragas_evaluator

def load_dataset(file_path: str) -> List[Dict[str, str]]:
    """Load questions and references from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate format
        if not isinstance(data, list):
            raise ValueError("Dataset must be a list of objects")
            
        for item in data:
            if "question" not in item:
                raise ValueError("Each item must have a 'question' field")
                
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation for RAG System")
    parser.add_argument("--dataset", type=str, default="starter_files/test_questions.json", help="Path to the dataset JSON file")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--chroma-db", type=str, help="Path to ChromaDB directory (optional, will discover if not provided)")
    parser.add_argument("--collection", type=str, help="ChromaDB collection name (optional)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="LLM model to use")
    
    args = parser.parse_args()
    
    # ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check API key
    openai_key = os.environ.get("CHROMA_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        print("Error: CHROMA_OPENAI_API_KEY or OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
        
    # Setup ChromaDB
    chroma_path = args.chroma_db
    collection_name = args.collection
    
    if not chroma_path or not collection_name:
        print("Discovering ChromaDB backends...")
        backends = rag_client.discover_chroma_backends()
        if not backends:
            print("No ChromaDB backends found. Please run the embedding pipeline first.")
            sys.exit(1)
            
        # Use the first available backend if not specified
        first_key = list(backends.keys())[0]
        backend = backends[first_key]
        
        if not chroma_path:
            chroma_path = backend["path"]
        if not collection_name:
            collection_name = backend["collection_name"]
            
        print(f"Using ChromaDB at {chroma_path} (Collection: {collection_name})")
    
    # Initialize RAG system
    try:
        collection = rag_client.initialize_rag_system(chroma_path, collection_name)
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        sys.exit(1)
        
    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    questions = load_dataset(args.dataset)
    print(f"Loaded {len(questions)} questions.")
    
    results = []
    
    # Evaluation Loop
    print("\nStarting evaluation...")
    for i, item in enumerate(questions):
        question = item["question"]
        reference = item.get("reference")
        
        print(f"[{i+1}/{len(questions)}] Processing: {question[:50]}...")
        
        # 1. Retrieve
        retrieval_results = rag_client.retrieve_documents(collection, question)
        
        if not retrieval_results or not retrieval_results['documents']:
            print("  Warning: No documents retrieved.")
            context_text = ""
            retrieved_contexts = []
        else:
            documents = retrieval_results['documents'][0]
            metadatas = retrieval_results['metadatas'][0] if 'metadatas' in retrieval_results else [{}] * len(documents)
            context_text = rag_client.format_context(documents, metadatas)
            retrieved_contexts = documents
            
        # 2. Generate
        answer = llm_client.generate_response(
            openai_key=openai_key,
            user_message=question,
            context=context_text,
            conversation_history=[], # No history for independent questions
            model=args.model
        )
        
        # 3. Evaluate
        metrics = ragas_evaluator.evaluate_response_quality(
            question=question,
            answer=answer,
            contexts=retrieved_contexts,
            reference=reference
        )
        
        # Store result
        result_entry = {
            "question": question,
            "answer": answer,
            "reference": reference,
            "contexts": retrieved_contexts,
            "metrics": metrics
        }
        
        # Flatten metrics for easier DataFrame handling
        if "error" not in metrics:
            for k, v in metrics.items():
                result_entry[k] = v
        
        results.append(result_entry)
        
    # Aggregate results
    print("\nCalculating aggregates...")
    
    # Identify metrics keys
    all_metric_keys = set()
    for r in results:
        if "metrics" in r and "error" not in r["metrics"] and isinstance(r["metrics"], dict):
             all_metric_keys.update(r["metrics"].keys())
             
    metric_cols = list(all_metric_keys)
    
    summary = {}
    if metric_cols:
        print("\nEvaluation Summary:")
        print("-" * 60)
        print(f"{'Metric':<25} | {'Mean':<10} | {'Min':<10} | {'Max':<10}")
        print("-" * 60)
        
        for metric in metric_cols:
            values = [r["metrics"][metric] for r in results if "metrics" in r and metric in r["metrics"] and isinstance(r["metrics"][metric], (int, float))]
            
            if values:
                mean_val = sum(values) / len(values)
                min_val = min(values)
                max_val = max(values)
                # Standard deviation
                std_val = 0.0
                if len(values) > 1:
                    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                    std_val = variance ** 0.5
                    
                summary[metric] = {
                    "mean": mean_val,
                    "min": min_val,
                    "max": max_val,
                    "std": std_val
                }
                print(f"{metric:<25} | {mean_val:.4f}     | {min_val:.4f}     | {max_val:.4f}")
            else:
                summary[metric] = {"note": "No valid numeric values"}
                
        print("-" * 60)
    else:
        print("\nNo metrics were calculated (possible errors in RAGAS evaluation).")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    summary_file = output_dir / f"summary_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()

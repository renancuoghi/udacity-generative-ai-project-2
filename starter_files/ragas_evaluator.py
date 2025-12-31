from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

import os

# RAGAS imports
try:
    from ragas import SingleTurnSample, EvaluationDataset
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str], reference: Optional[str] = None) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics
    
    Args:
        question: The user's question
        answer: The generated answer from the RAG system
        contexts: List of retrieved context documents
        reference: Optional reference answer for precision metrics
    
    Returns:
        Dictionary containing evaluation scores for each metric
    """
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    try:

        # Get API key
        api_key = os.environ.get("CHROMA_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = None
        if api_key and api_key.startswith("voc-"):
            base_url = "https://openai.vocareum.com/v1"

        # Create evaluator LLM with model gpt-3.5-turbo
        llm_kwargs = {"model": "gpt-3.5-turbo", "api_key": api_key}
        if base_url:
            llm_kwargs["base_url"] = base_url
            
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(**llm_kwargs))
        
        # Create evaluator_embeddings with model text-embedding-3-small
        emb_kwargs = {"model": "text-embedding-3-small", "api_key": api_key}
        if base_url:
            emb_kwargs["base_url"] = base_url
            
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(**emb_kwargs))
        
        # Define an instance for each metric to evaluate
        # Non-LLM metrics (faster, no API calls needed)
        bleu_score = BleuScore()
        rouge_score = RougeScore()
        
        # LLM-based metrics (require LLM and embeddings)
        response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        faithfulness = Faithfulness(llm=evaluator_llm)
        
        # Create a list of metrics to evaluate
        metrics = [bleu_score, rouge_score, response_relevancy, faithfulness]
        

        
        # Create a SingleTurnSample for evaluation
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference if reference else ""
        )
        
        # Evaluate the response using the metrics
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        # Convert result to dictionary and return the evaluation results
        # The result is a pandas DataFrame-like object
        # result.to_pandas() returns a DataFrame
        df = result.to_pandas()
        scores = df.to_dict('records')[0] if not df.empty else {}
        
        # Return only the metric scores (filter out input fields)
        metric_scores = {
            key: value for key, value in scores.items() 
            if key not in ['user_input', 'response', 'retrieved_contexts', 'reference']
        }
        
        return metric_scores
        
    except Exception as e:
        # Handle any errors during evaluation gracefully
        return {"error": f"Evaluation failed: {str(e)}"}

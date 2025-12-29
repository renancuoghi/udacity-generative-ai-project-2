from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
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
        # Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
        
        # Create evaluator_embeddings with model text-embedding-3-small
        evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
        
        # Define an instance for each metric to evaluate
        # Non-LLM metrics (faster, no API calls needed)
        bleu_score = BleuScore()
        rouge_score = RougeScore()
        
        # LLM-based metrics (require LLM and embeddings)
        response_relevancy = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        faithfulness = Faithfulness(llm=evaluator_llm)
        
        # Create a list of metrics to evaluate
        metrics = [bleu_score, rouge_score, response_relevancy, faithfulness]
        
        # Add context precision metric if reference answer is provided
        if reference:
            context_precision = NonLLMContextPrecisionWithReference()
            metrics.append(context_precision)
        
        # Create a SingleTurnSample for evaluation
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=reference if reference else ""
        )
        
        # Evaluate the response using the metrics
        result = evaluate(
            dataset=[sample],
            metrics=metrics
        )
        
        # Convert result to dictionary and return the evaluation results
        # The result is a pandas DataFrame, so we extract the first row as a dict
        scores = result.to_dict('records')[0] if len(result) > 0 else {}
        
        # Return only the metric scores (filter out input fields)
        metric_scores = {
            key: value for key, value in scores.items() 
            if key not in ['user_input', 'response', 'retrieved_contexts', 'reference']
        }
        
        return metric_scores
        
    except Exception as e:
        # Handle any errors during evaluation gracefully
        return {"error": f"Evaluation failed: {str(e)}"}

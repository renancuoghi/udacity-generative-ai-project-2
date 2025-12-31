from typing import Dict, List
from openai import OpenAI

def generate_response(openai_key: str, user_message: str, context: str, 
                      conversation_history: List[Dict], model: str = "gpt-3.5-turbo",
                      max_history: int = 5) -> str:
    """
    Generate response using OpenAI with context.
    
    Args:
        openai_key: OpenAI API key
        user_message: The current user message
        context: Retrieved context documents
        conversation_history: List of past conversation messages
        model: Model to use for generation
        max_history: Maximum number of past conversation turns to include
    """

    # Create OpenAI Client
    if openai_key.startswith("voc-"):
        client = OpenAI(
             base_url="https://openai.vocareum.com/v1",
             api_key=openai_key
         )
    else:
        client = OpenAI(api_key=openai_key)

    # Define system prompt
    system_prompt = (
        "You are an expert NASA mission specialist. Your goal is to provide accurate, "
        "technical, and helpful information about NASA missions based on the provided context. "
        "Use the provided context to answer the user's question. "
        "If the answer is not in the context, say you don't know. "
        "Always cite your sources using the [Source #i] format provided in the context."
    )

    # Prepare messages
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add chat history (pruned to max_history)
    # Expecting conversation_history to be a list of dicts like {"role": "user", "content": "..."}
    recent_history = conversation_history[-max_history:] if max_history > 0 else []
    for conversation in recent_history:
        messages.append(conversation)
    
    # Add current user message with context
    final_user_message = f"Context: {context}\n\nQuestion: {user_message}"
    messages.append({"role": "user", "content": final_user_message})

    # Send request to OpenAI
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"
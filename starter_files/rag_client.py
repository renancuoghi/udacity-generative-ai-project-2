import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path
from openai import OpenAI
import os

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # Scanning for directories that likely contain ChromaDB
    potential_dirs = [d for d in current_dir.iterdir() if d.is_dir() and ("chroma" in d.name or "_db" in d.name)]

    # Loop through each discovered directory
    for db_dir in potential_dirs:
        try:
            # Initialize database client with directory path and configuration settings
            # We use a try-except block to verify if it's a valid ChromaDB
            client = chromadb.PersistentClient(path=str(db_dir))
            
            # Retrieve list of available collections from the database
            collections = client.list_collections()
            
            # Loop through each collection found
            for col in collections:
                # Create unique identifier key combining directory and collection names
                key = f"{db_dir.name}/{col.name}"
                
                # Build information dictionary
                backends[key] = {
                    "path": str(db_dir),
                    "collection_name": col.name,
                    "display_name": f"{col.name} ({db_dir.name})",
                    "count": col.count()
                }
        
        except Exception as e:
            # Handle connection or access errors gracefully
            print(f"Error accessing likely DB directory {db_dir}: {e}")
            # We don't add broken backends to the list

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    # Create a chomadb persistentclient
    client = chromadb.PersistentClient(path=chroma_dir)
    # Return the collection with the collection_name
    return client.get_collection(collection_name)

def get_query_embedding(openai_key: str, query: str, model: str = "text-embedding-3-small") -> List[float]:
    """Generate OpenAI embedding for a query string"""
    
    # Initialize OpenAI client
    if openai_key.startswith("voc-"):
         client = OpenAI(
             base_url="https://openai.vocareum.com/v1",
             api_key=openai_key
         )
    else:
         client = OpenAI(api_key=openai_key)
         
    try:
        response = client.embeddings.create(
            model=model,
            input=query
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Get OpenAI API key from environment variable
    openai_key = os.environ.get("CHROMA_OPENAI_API_KEY")
    if not openai_key:
        print("Error: CHROMA_OPENAI_API_KEY environment variable not set.")
        return None

    # Generate embedding for the query
    query_embedding = get_query_embedding(openai_key, query)
    
    if not query_embedding:
        print("Failed to generate query embedding. Returning None.")
        return None

    # Initialize filter variable to None (represents no filtering)
    where_clause = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # If filter conditions are met, create filter dictionary
        where_clause = {"mission": mission_filter}

    # Execute database query
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_clause
    )

    # Inject IDs and distances into metadata for deduplication and sorting in format_context
    if results and "ids" in results and "metadatas" in results:
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            meta["_id"] = results["ids"][0][i]
            if "distances" in results:
                meta["_distance"] = results["distances"][0][i]

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context with deduplication and sorting via metadata"""
    if not documents:
        return ""
    
    # Combine documents and metadatas for sorting/deduplication
    combined = list(zip(documents, metadatas))
    
    # Deduplicate by injected ID or by text if ID is missing
    seen_ids = set()
    seen_texts = set()
    unique_combined = []
    
    for doc, meta in combined:
        doc_id = meta.get("_id")
        if doc_id:
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_combined.append((doc, meta))
        else:
            # Fallback to text hash if ID is somehow missing
            text_hash = hash(doc)
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_combined.append((doc, meta))
    
    # Sort by distance (lower is better for cosine distance)
    # distance is injected as _distance
    unique_combined.sort(key=lambda x: x[1].get("_distance", 0.0))
    
    # Initialize list with header text for context section
    context_parts = []
    
    for i, (doc, meta) in enumerate(unique_combined, 1):
        # Extract mission information from metadata with fallback value
        mission = meta.get("mission", "Unknown Mission")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()
        
        # Extract source information from metadata with fallback value
        source_path = meta.get("source", "Unknown Source")
        source_name = Path(source_path).name
        
        # Create formatted source header with index number and extracted information
        header = f"[Source #{i}] Mission: {mission} | Source: {source_name}"
        
        # Add source header to context parts list
        context_parts.append(header)
        
        # Check document length and truncate if necessary (e.g., 500 chars for brevity in display if needed, but RAG usually wants full chunks)
        # Using full content as starter code didn't specify strict limit, but typical RAG contexts are chunked already.
        context_parts.append(doc)
        context_parts.append("") # Empty line for separation

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)
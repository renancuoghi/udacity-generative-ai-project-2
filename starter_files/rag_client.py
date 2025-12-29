import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path

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

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_clause = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    if mission_filter and mission_filter.lower() != "all":
        # If filter conditions are met, create filter dictionary
        where_clause = {"mission": mission_filter}

    # Execute database query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause
    )

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # Initialize list with header text for context section
    context_parts = []

    # Loop through paired documents and their metadata using enumeration
    # documents[0] is the list of documents for the first query (we typically query one at a time)
    # But the input signature says List[str], assuming it is the flattened list of docs.
    # However, Chroma returns a list of lists. Let's assume the caller passes the inner list or we handle it?
    # Based on type hint List[str], we assume it's a list of strings.
    
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
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
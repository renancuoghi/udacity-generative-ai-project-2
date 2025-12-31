
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add starter_files to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../starter_files')))

# Mock chromadb and openai before importing rag_client
mock_chromadb = MagicMock()
mock_chromadb_config = MagicMock()
mock_openai_lib = MagicMock()
sys.modules['chromadb'] = mock_chromadb
sys.modules['chromadb.config'] = mock_chromadb_config
sys.modules['openai'] = mock_openai_lib

import rag_client

class TestRAGClient(unittest.TestCase):
    
    @patch('rag_client.OpenAI')
    def test_retrieve_documents_calls_query_with_embeddings(self, mock_openai):
        # Setup mock OpenAI client
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_embedding_response
        
        # Setup mock ChromaDB collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["doc1"]], 
            "metadatas": [[{"mission": "apollo11"}]],
            "distances": [[0.1]]
        }
        
        # Call the function
        with patch.dict('os.environ', {'CHROMA_OPENAI_API_KEY': 'sk-test'}):
            query = "What was Apollo 11?"
            results = rag_client.retrieve_documents(mock_collection, query)
        
        # Assertions
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=query
        )
        
        mock_collection.query.assert_called_once()
        args, kwargs = mock_collection.query.call_args
        self.assertIn('query_embeddings', kwargs)
        self.assertEqual(kwargs['query_embeddings'], [[0.1, 0.2, 0.3]])
        
        # Verify injection
        self.assertEqual(results["metadatas"][0][0]["_id"], "id1")
        self.assertEqual(results["metadatas"][0][0]["_distance"], 0.1)

    def test_format_context_metadata_injection(self):
        # Setup test data with injected metadata
        documents = ["doc1", "doc2", "doc1", "doc3"]
        metadatas = [
            {"mission": "apollo_11", "source": "file1", "_id": "id1", "_distance": 0.1},
            {"mission": "apollo_13", "source": "file2", "_id": "id2", "_distance": 0.5},
            {"mission": "apollo_11", "source": "file1", "_id": "id1", "_distance": 0.1},
            {"mission": "challenger", "source": "file3", "_id": "id3", "_distance": 0.2}
        ]
        
        # Call the function
        context = rag_client.format_context(documents, metadatas)
        
        # Assertions
        # 1. Deduplication: "doc1" should only appear once.
        # 2. Sorting: distances are [0.1, 0.5, 0.2] -> sorted should be 0.1 (id1), 0.2 (id3), 0.5 (id2)
        
        # Check order and absence of duplicate
        lines = context.split('\n')
        headers = [line for line in lines if line.startswith("[Source #")]
        
        self.assertEqual(len(headers), 3)
        self.assertIn("Mission: Apollo 11", headers[0])
        self.assertIn("Mission: Challenger", headers[1])
        self.assertIn("Mission: Apollo 13", headers[2])
        
        self.assertIn("doc1", context)
        self.assertIn("doc2", context)
        self.assertIn("doc3", context)

if __name__ == '__main__':
    unittest.main()

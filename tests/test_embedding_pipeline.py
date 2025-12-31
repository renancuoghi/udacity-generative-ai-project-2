import unittest
import sys
import os
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Add the project root to sys.path to allow importing from starter_files
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from starter_files.embedding_pipeline import ChromaEmbeddingPipelineTextOnly

class TestEmbeddingPipeline(unittest.TestCase):

    def setUp(self):
        # Mock OpenAI and ChromaDB to avoid real API calls and disk IO
        self.patcher_openai = patch('starter_files.embedding_pipeline.OpenAI')
        self.patcher_chromadb = patch('starter_files.embedding_pipeline.chromadb.PersistentClient')
        
        self.mock_openai_class = self.patcher_openai.start()
        self.mock_chroma_class = self.patcher_chromadb.start()
        
        self.mock_openai_client = MagicMock()
        self.mock_openai_class.return_value = self.mock_openai_client
        
        self.mock_chroma_client = MagicMock()
        self.mock_chroma_class.return_value = self.mock_chroma_client
        
        self.mock_collection = MagicMock()
        self.mock_chroma_client.get_or_create_collection.return_value = self.mock_collection
        
        self.pipeline = ChromaEmbeddingPipelineTextOnly(
            openai_api_key="test-key",
            chroma_persist_directory="./test_db",
            collection_name="test_collection",
            chunk_size=100,
            chunk_overlap=20
        )

    def tearDown(self):
        self.patcher_openai.stop()
        self.patcher_chromadb.stop()

    def test_chunk_text_basic(self):
        text = "This is a test sentence. This is another test sentence."
        metadata = {"source": "test"}
        chunks = self.pipeline.chunk_text(text, metadata)
        
        self.assertTrue(len(chunks) >= 1)
        self.assertEqual(chunks[0][1]['source'], "test")
        self.assertEqual(chunks[0][1]['total_chunks'], len(chunks))

    def test_chunk_text_with_overlap(self):
        # Create text that will definitely be chunked
        text = "A" * 150
        metadata = {"source": "test"}
        chunks = self.pipeline.chunk_text(text, metadata)
        
        self.assertGreater(len(chunks), 1)
        # Check if overlap works (part of 1st chunk in 2nd chunk)
        # With chunk_size=100, overlap=20
        # 1st chunk: index 0 to ~100
        # 2nd chunk start: 100 - 20 = 80
        pass 

    def test_extract_mission_from_path(self):
        self.assertEqual(self.pipeline.extract_mission_from_path(Path("data/apollo11/doc.txt")), "apollo_11")
        self.assertEqual(self.pipeline.extract_mission_from_path(Path("data/apollo13/doc.txt")), "apollo_13")
        self.assertEqual(self.pipeline.extract_mission_from_path(Path("data/challenger/doc.txt")), "challenger")
        self.assertEqual(self.pipeline.extract_mission_from_path(Path("data/other/doc.txt")), "unknown")

    def test_extract_data_type_from_path(self):
        self.assertEqual(self.pipeline.extract_data_type_from_path(Path("data/transcripts/doc.txt")), "transcript")
        self.assertEqual(self.pipeline.extract_data_type_from_path(Path("data/textract/doc.txt")), "textract_extracted")

    @patch('starter_files.embedding_pipeline.ChromaEmbeddingPipelineTextOnly.get_embedding')
    @patch('starter_files.embedding_pipeline.ChromaEmbeddingPipelineTextOnly.check_document_exists')
    def test_add_documents_skip_mode(self, mock_exists, mock_get_embedding):
        mock_exists.return_value = True
        documents = [("text", {"chunk_index": 0, "source": "src", "mission": "m"})]
        
        stats = self.pipeline.add_documents_to_collection(documents, Path("test.txt"), update_mode='skip')
        
        self.assertEqual(stats['skipped'], 1)
        self.assertEqual(stats['added'], 0)
        self.mock_collection.upsert.assert_not_called()

    @patch('starter_files.embedding_pipeline.ChromaEmbeddingPipelineTextOnly.get_embedding')
    @patch('starter_files.embedding_pipeline.ChromaEmbeddingPipelineTextOnly.check_document_exists')
    def test_add_documents_update_mode(self, mock_exists, mock_get_embedding):
        mock_exists.return_value = True
        mock_get_embedding.return_value = [0.1, 0.2]
        documents = [("text", {"chunk_index": 0, "source": "src", "mission": "m"})]
        
        stats = self.pipeline.add_documents_to_collection(documents, Path("test.txt"), update_mode='update')
        
        self.assertEqual(stats['updated'], 1)
        self.mock_collection.upsert.assert_called_once()

    @patch('starter_files.embedding_pipeline.ChromaEmbeddingPipelineTextOnly.get_embedding')
    def test_add_documents_replace_mode(self, mock_get_embedding):
        mock_get_embedding.return_value = [0.1, 0.2]
        documents = [("text", {"chunk_index": 0, "source": "src", "mission": "m"})]
        
        stats = self.pipeline.add_documents_to_collection(documents, Path("test.txt"), update_mode='replace')
        
        # Verify deletion was called
        # self.pipeline.collection.delete is self.mock_collection.delete
        self.mock_collection.delete.assert_called_once()
        self.assertEqual(stats['added'], 1)
        self.mock_collection.upsert.assert_called_once()

if __name__ == '__main__':
    unittest.main()

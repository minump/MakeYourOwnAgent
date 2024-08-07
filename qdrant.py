#QDrant vector store

import pandas as pd
from typing import Any, List
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from data_loaders import DataLoaders

from config import Config
import uuid
import sys
import logging
import os
from dotenv import load_dotenv

# create and configure logger
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%dT%H:%M:%S',
                    format='%(asctime)-15s.%(msecs)03dZ %(levelname)-7s : %(name)s - %(message)s',
                    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler(sys.stdout)])
# create log object with current module name
log = logging.getLogger(__name__)

# Load .env file
load_dotenv()

class Qdrantdb:
    """
    Qdrant client for Qdrant server hosted in QDRANT_URL
    """
    def __init__(self) -> None:
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', "email-collection")
        self.qdrant_url = os.getenv('QDRANT_URL', "localhost")
        self.qdrant_client = QdrantClient(url=self.qdrant_url, port=6333, timeout=300)  # timeout=300
        self.embeddings = OpenAIEmbeddings()

    def get_qdrant_client(self):
        return self.qdrant_client
    
    def collection_exists(self) -> bool:
        return self.qdrant_client.collection_exists(self.collection_name)
    
    def create_collection(self) -> bool:
        status = self.qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(distance=Distance.COSINE, size=1536),
            )
        return status
    
    def get_collection_count(self) -> int:
        return self.qdrant_client.count(collection_name=self.collection_name)

    def upsert_data(self, data_dir_path: str) -> bool:
        """
        Method to upload data to qdrant collection
        """
        # load data
        data_loader = DataLoaders(data_dir_path=data_dir_path)
        log.info("Loading files from directory %s", data_dir_path)
        dir_csv_loader = data_loader.csv_loader()
        dir_word_loader = data_loader.word_loader()
        dir_pdf_loader = data_loader.pdf_loader()
        csv_data = dir_csv_loader.load()
        word_data = dir_word_loader.load()
        pdf_data = dir_pdf_loader.load()
        texts , metadatas = DataLoaders.get_text_metadatas(csv_data, pdf_data, word_data)
        
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=1000,
                chunk_overlap=150,
                separators=[
                    "\n\n", "\n", ". ", " ", ""
                ]  # try to split on paragraphs... fallback to sentences, then chars, ensure we always fit in context window
            )
        log.info("Creating documents from text splitter")
        docs: List[Document] = text_splitter.create_documents(texts=texts, metadatas=metadatas)  # gives a Document class with attributes page_content and metadata
        log.info("Creating text embeddings")
        text_embeddings = self.embeddings.embed_documents([doc.page_content for doc in docs])
        #text_embeddings_dict = [{texts[i]: text_embeddings[i]} for i in range(len(texts))]
        payload = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        # Qdrant bulk upsert
        vectors: list[PointStruct] = []
        log.info("Convert to vector")
        vectors = [PointStruct(id=str(uuid.uuid4()), vector=text_embeddings[i], payload=payload[i]) for i in range(len(docs))]
        
        try:
            log.info("Qdrant bulk upload of vector points")
            self.qdrant_client.upload_points(collection_name=self.collection_name, points=vectors)
            return True
        except Exception as e:
            log.error(f"Failed to upsert data into collection: {self.collection_name}. Error: {e}")
            if hasattr(e, 'message'):
                log.error(f"Message: {e.message} ")  
            elif hasattr(e, 'response'):
                log.error(f"Status code: {e.response.status_code}, Body: {e.response.body}")  # 
            return False
        
    def vector_store(self):
        """
        Method to return a qdrant vector store
        """
        # create vector Store
        vector_store = Qdrant(client=self.qdrant_client,
                        embeddings=self.embeddings,
                        collection_name=self.collection_name,
                        metadata_payload_key="metadata",
                        content_payload_key="content"
                    )
        return vector_store


    def vector_search(self, user_query):
        user_query_embedding = self.embeddings.embed_query(user_query)
        top_n = 50 # return n closest points
        results = self.qdrant_client.search(collection_name=self.collection_name,
                                            query_vector=user_query_embedding,
                                             limit=top_n,
                                              with_vectors=False )
        return results
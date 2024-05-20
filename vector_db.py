# Vectore Store using Qdrant

import pandas as pd
from typing import Any, List
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader, DataFrameLoader, PyPDFLoader, UnstructuredRSTLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import Config
import torch
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



    

def create_local_vector_store(data_dir_path: str):
    """
    Method to create a qdrant vectore store
    """
    
    texts, metadatas = get_data(data_dir_path)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n", "\n", ". ", " ", ""
            ]  # try to split on paragraphs... fallback to sentences, then chars, ensure we always fit in context window
        )

    docs: List[Document] = text_splitter.create_documents(texts=texts, metadatas=metadatas)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', "data-collection")

    # create vector Store
    vector_store = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name=collection_name,
        )

    return vector_store

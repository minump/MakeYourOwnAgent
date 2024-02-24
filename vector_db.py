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

def get_data(data_dir_path: str):
    """
    Method to get data using data loaders
    Returns:
    texts : List[str]
    metadatas: List[str]
    """
    # load data 
    log.info("Loading files from directory %s", data_dir_path)
    csv_loader_kwargs = {
                        "csv_args":{
                            "delimiter": ",",
                            "quotechar": '"',
                            },
                        }
    dir_csv_loader = DirectoryLoader(data_dir_path, glob="**/*.csv", use_multithreading=True,
                                loader_cls=CSVLoader, 
                                loader_kwargs=csv_loader_kwargs,
                                )
    dir_pdf_loader = DirectoryLoader(data_dir_path, glob="**/*.pdf",
                                     loader_cls=PyPDFLoader,
                                    )
    dir_rst_loader = DirectoryLoader(data_dir_path, glob="**/*.rst",
                                     loader_cls=UnstructuredRSTLoader,
                                    )
    csv_data = dir_csv_loader.load()
    pdf_data = dir_pdf_loader.load()
    rst_data = dir_rst_loader.load()
    csv_texts = [doc.page_content for doc in csv_data]
    csv_metadatas = [doc.metadata for doc in csv_data]
    pdf_texts = [doc.page_content for doc in pdf_data]
    pdf_metadatas = [doc.metadata for doc in pdf_data]  # metadata={'source': 'data/select_data/The SHIELD Story_June2023.pdf', 'page': 8}
    rst_texts = [doc.page_content for doc in rst_data]
    rst_metadatas = [doc.metadata for doc in rst_data]
    texts = csv_texts + pdf_texts + rst_texts
    metadatas = csv_metadatas + pdf_metadatas + rst_metadatas

    return texts, metadatas
    

def create_vector_store(data_dir_path: str):
    """
    Method to create a qdrant vectore store
    """
    
    texts, metadatas = get_data(data_dir_path)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000,
            chunk_overlap=150,
            separators=[
                "\n\n", "\n", ". ", " ", ""
            ]  # try to split on paragraphs... fallback to sentences, then chars, ensure we always fit in context window
        )

    docs: List[Document] = text_splitter.create_documents(texts=texts, metadatas=metadatas)

    embeddings = Config.embeddings
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', "data-collection")

    # create vector Store
    vector_store = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name=collection_name,
        )

    return vector_store

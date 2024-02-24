# Create a RAG LLM using Qdrant vector store. Read multiple user queries and return agent output

import pandas as pd
from typing import Any, List
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from config import Config
from vector_db import create_vector_store

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


def create_rag_agent(data_dir_path):
    """
    Create a RAG agent
    """
    vector_store = create_vector_store(data_dir_path)
    llm = Config.llm

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever())
    
    return qa

    

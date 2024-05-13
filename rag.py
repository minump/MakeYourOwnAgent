# Create a RAG LLM using Qdrant vector store. Read multiple user queries and return output

import pandas as pd
from typing import Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from qdrant import Qdrantdb
from vector_db import create_local_vector_store

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


def rag_local_vectorstore(data_dir_path):
    """
    Create a RAG agent
    """
    vector_store = create_local_vector_store(data_dir_path)
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever())
    
    return qa


qdrantdb = Qdrantdb()


class CustomRetriever(BaseRetriever):
    """
    Custom Retriver class
    """

    def get_similar_docs(self, user_query:str):
        """
        Method to get top similar docs from vector store
        """
        try:
            # Retrieve similar documents based on the user input vector
            results = qdrantdb.vector_search(user_query=user_query)
            log.info("Retrieved similar documents successfully.")
            return results
        except Exception as e:
            log.error(f"Failed to retrieve similar documents: {str(e)}")
            return []

    def _get_relevant_documents(self, user_query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        Method to get relevant documents
        """
        log.info("Get relevent docs for query: %s", user_query)
        search_docs = self.get_similar_docs(user_query=user_query) 
        found_docs = []
        for doc in search_docs:
            try:
                payload = doc.payload  # {'content': 'str', 'metadata': {'row': 69, 'source': 'communication_channel_2_thread_multi_3.csv'}}
                doc_content = payload["content"]
                doc_metadata = payload["metadata"]
                if "source" in doc_metadata:
                    source_filename = doc_metadata['source'].split('/')[-1]
                    doc_metadata['source'] = source_filename
                found_docs.append(Document(page_content=doc_content, metadata=doc_metadata))
            except Exception as e:
                log.error(f"Error in rag search : {str(e)}")

        return found_docs



def create_retriever(vector_store):
    """
    Create a RAG QA retriver
    """

    llm: ChatOpenAI = ChatOpenAI(
            temperature=0,
            model="gpt-4-0125-preview",
            max_retries=500,
            )
    retriever = vector_store.as_retriever()  # search_type="mmr"
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever)
    
    return qa


def create_custom_retriever():
    """
    Create a custom RAG QA retriever
    """
    qa_prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If the question is to request links, please only return the source links with no answer.
    2. If you don't know the answer, don't try to make up an answer. Just say **I can't find the final answer but you may want to check the following links** and add the source links as a list.
    3. If you find the answer, write the answer in a concise way and add the list of sources that are **directly** used to derive the answer. Exclude the sources that are irrelevant to the final answer.

    {context}

    Question: {question}
    Helpful Answer:"""
    qa_chain_prompt = PromptTemplate.from_template(qa_prompt_template)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source", "page", "row"],
        template="Context:\ncontent:{page_content}\nsource:{source} row or page:{row_page}",
    )

    retriever = CustomRetriever()
    llm: ChatOpenAI = ChatOpenAI(
            temperature=0,
            model="gpt-4-0125-preview",
            max_retries=500,
        )
    llm_chain = LLMChain(llm=llm, prompt=qa_chain_prompt, callbacks=None, verbose=True)
    combine_documents_chain = StuffDocumentsChain(
                                llm_chain=llm_chain,
                                document_variable_name="context",
                                document_prompt=document_prompt,
                                callbacks=None,
                            )
    qa = RetrievalQA(
            combine_documents_chain=combine_documents_chain,
            callbacks=None,
            verbose=False,
            retriever=retriever,
            return_source_documents=True,
        )
    return qa

    
def rag_results(vector_store, queries: List[str]):
    """
    Call retriever for user queries
    """
    retriever = create_retriever(vector_store=vector_store)
    log.info("Calling llm with batch queries")
    responses = retriever.batch(queries)
    log.info("QA retrieval completed")
    return responses


def rag_search_results(queries: List[str]):
    """
    RAG method
    Get similar documents form vector search
    Give the documents and user query as context to LLM
    Return answers with source citations
    """
    retriever = create_custom_retriever()
    log.info("Calling llm with batch queries")
    responses = retriever.batch(queries)
    log.info("QA retrieval completed")
    return responses



    

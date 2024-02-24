# Configurations to use

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
# Load .env file
load_dotenv()


class Config:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
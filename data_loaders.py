# Data Loaders from Langchain

from dotenv import load_dotenv
import logging
from langchain_community.document_loaders import CSVLoader, DataFrameLoader, PyPDFLoader, Docx2txtLoader, UnstructuredRSTLoader, DirectoryLoader


# Load .env file
load_dotenv(override=True)

# create and configure logger
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%dT%H:%M:%S',
                    format='%(asctime)-15s.%(msecs)03dZ %(levelname)-7s : %(name)s - %(message)s',
                    handlers=[logging.FileHandler("llm.log"), logging.StreamHandler(sys.stdout)])
# create log object with current module name
log = logging.getLogger(__name__)


class DataLoaders:
    """
    specify all data loaders here
    """
    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
    
    def csv_loader(self):
        csv_loader_kwargs = {
                            "csv_args":{
                                "delimiter": ",",
                                "quotechar": '"',
                                },
                            }
        dir_csv_loader = DirectoryLoader(self.data_dir_path, glob="**/*.csv", use_multithreading=True,
                                    loader_cls=CSVLoader, 
                                    loader_kwargs=csv_loader_kwargs,
                                    )
        return dir_csv_loader
    
    def pdf_loader(self):
        dir_pdf_loader = DirectoryLoader(self.data_dir_path, glob="**/*.pdf",
                                    loader_cls=PyPDFLoader,
                                    )
        return dir_pdf_loader
    
    def word_loader(self):
        dir_word_loader = DirectoryLoader(self.data_dir_path, glob="**/*.docx",
                                    loader_cls=Docx2txtLoader,
                                    )
        return dir_word_loader
    
    def rst_loader(self):
        rst_loader_kwargs = {
                        "mode":"single"
                        }
        dir_rst_loader = DirectoryLoader(self.data_dir_path, glob="**/*.rst",
                                    loader_cls=UnstructuredRSTLoader, 
                                    loader_kwargs=rst_loader_kwargs
                                    )
        return dir_rst_loader

    def get_text_metadatas(csv_data=None, pdf_data=None, word_data=None, rst_data=None):
        """
        Format text and metadata content
        """
        csv_texts = [doc.page_content for doc in csv_data]
        csv_metadatas = [{'source': doc.metadata['source'], 'row_page': doc.metadata['row']} for doc in csv_data] # metadata={'source': 'filename.csv', 'row': 0}
        pdf_texts = [doc.page_content for doc in pdf_data]
        pdf_metadatas = [{'source': doc.metadata['source'], 'row_page': doc.metadata['page']} for doc in pdf_data]  # metadata={'source': 'data/filename.pdf', 'page': 8}
        word_texts = [doc.page_content for doc in word_data]
        word_metadatas = [{'source': doc.metadata['source'], 'row_page': ''} for doc in word_data] 
        rst_texts = [doc.page_content for doc in rst_data]
        rst_metadatas = [doc.metadata for doc in rst_data]
        texts = csv_texts + pdf_texts + word_texts + rst_texts
        metadatas = csv_metadatas + pdf_metadatas + word_metadatas + rst_metadatas
        return texts, metadatas

    

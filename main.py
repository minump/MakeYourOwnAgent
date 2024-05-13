# Main file to run

import sys
import logging
import os
from dotenv import load_dotenv
from rag import create_rag_agent

import pypandoc
pypandoc.download_pandoc()

# create and configure logger
logging.basicConfig(level=logging.INFO, datefmt='%Y-%m-%dT%H:%M:%S',
                    format='%(asctime)-15s.%(msecs)03dZ %(levelname)-7s : %(name)s - %(message)s',
                    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler(sys.stdout)])
# create log object with current module name
log = logging.getLogger(__name__)

# Load .env file
load_dotenv()


def main():
    """
    Create a RAG agent that has knowledge from data directory and responds to user queries from query directory
    """
    data_dir = os.getenv('DATA_DIR_PATH', "data")
    query_dir = os.getenv('QUERY_DIR_PATH', "query")
    queries = []

    agent = create_rag_agent(data_dir)

    # read text files in query directory
    if os.path.isdir(query_dir):
        for file_name in os.listdir(query_dir):
            if file_name.endswith('.txt'):
                file_path = os.path.join(query_dir, file_name)
                with open(file_path, 'r') as file:
                    query = file.read().strip()
                    queries.append({"query": query})
    else:
        log.error("%s is not a valid directory", query_dir)

    log.info("Calling agent with batch queries")
    responses = agent.batch(queries)
    for response in responses:
        log.info("User query : %s", response['query'])
        log.info("RAG output : %s", response['result'])
        log.info('=' * 25)

    pass



if __name__ == '__main__':
    main()
    
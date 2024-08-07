# Dockerfile for jupyter notebook HandsOn RAG session

FROM python:3.12-slim

# Install necessary packages for building Python packages
RUN apt-get update && apt-get install -y gcc python3-dev
# RUN apk add --no-cache gcc g++ python3-dev musl-dev linux-headers
# RUN apt-get install -y pandoc
RUN apt-get install -y pandoc

WORKDIR /src

COPY delta_docs /src/delta_docs
COPY requirements.txt .env /src

RUN pip install -r requirements.txt

COPY rag.ipynb /src

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]

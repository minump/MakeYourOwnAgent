FROM jupyter/base-notebook

WORKDIR /src

COPY delta_docs /src/delta_docs
COPY requirements.txt .env /src

RUN pip install -r requirements.txt

COPY rag.ipynb /src

CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]

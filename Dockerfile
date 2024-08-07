FROM jupyter/base-notebook

LABEL maintainer="Minu Mathew <minum@illinois.edu>"
ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER root
COPY requirements.txt ./
RUN pip install -r requirements.txt

USER 1000
COPY .env ./
COPY delta_docs ./delta_docs
COPY rag.ipynb ./

# CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]

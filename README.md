## RAG Pipeline using FastAPI
A Retrieval-Augmented Generation (RAG) backend built with FastAPI.
This project performs PDF ingestion, text chunking, embedding generation, FAISS vector storage, and semantic similarity search via a REST API.

## Features:

 PDF document ingestion

 Recursive text chunking

 HuggingFace sentence-transformer embeddings

 FAISS vector database for similarity search

 Semantic retrieval based on user query

 FastAPI REST API

 Swagger UI documentation

## Project Structure
 RAG-Pipeline/
│
├── backend/
│   ├── main.py
│   ├── rag_pipeline.py
│   └── __init__.py
│
├── data/
│   └── sample_pdf.pdf
│
├── requirements.txt
└── .gitignore


## How it works?

Load PDF documents

Split documents into smaller chunks

Convert each chunk into embeddings

Store embeddings in FAISS vector store

Accept user query via API

Retrieve top relevant chunks

Return contextual information

## Tech Stack

Python 3.13

FastAPI

LangChain

HuggingFace Sentence Transformers

FAISS

Uvicorn

import os
from typing import List

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load environment variables
load_dotenv()


class RAGPipeline:
    def __init__(self, data_path="data", index_path="faiss_index"):
        self.data_path = data_path
        self.index_path = index_path

        # Embedding model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.vectorstore = None
        self._initialize_vectorstore()

        # 🔥 LLM Initialization (Groq)
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-8b-8192"
        )

    # ------------------------------
    # Load PDF Documents
    # ------------------------------
    def _load_documents(self) -> List[Document]:
        documents = []
        for file in os.listdir(self.data_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.data_path, file))
                documents.extend(loader.load())
        return documents

    # ------------------------------
    # Initialize FAISS Vector Store
    # ------------------------------
    def _initialize_vectorstore(self):
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            docs = self._load_documents()
            split_docs = self.text_splitter.split_documents(docs)

            self.vectorstore = FAISS.from_documents(
                split_docs,
                self.embeddings
            )

            self.vectorstore.save_local(self.index_path)

    # ------------------------------
    # Retrieve Context Only
    # ------------------------------
    def build_context(self, query: str, k: int = 4) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])

    # ------------------------------
    # Generate Answer Using LLM
    # ------------------------------
    def generate_answer(self, query: str, k: int = 4) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_template(
            """
            You are an AI assistant.
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {question}
            """
        )

        chain = prompt | self.llm

        response = chain.invoke({
            "context": context,
            "question": query
        })

        return response.content
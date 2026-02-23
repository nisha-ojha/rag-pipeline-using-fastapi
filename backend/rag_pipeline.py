import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.documents import Document

class RAGPipeline:
    def __init__(self, data_path="data", index_path="faiss_index"):
        self.data_path = data_path
        self.index_path = index_path

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        self.vectorstore = None
        self._initialize_vectorstore()

    def _load_documents(self) -> List[Document]:
        documents = []
        for file in os.listdir(self.data_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.data_path, file))
                documents.extend(loader.load())
        return documents

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

    def build_context(self, query: str, k: int = 4) -> str:
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n\n".join([doc.page_content for doc in docs])
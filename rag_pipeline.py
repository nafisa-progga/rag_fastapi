# rag_pipeline.py

import os
from dotenv import load_dotenv

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Load environment variables once for the entire application
load_dotenv()

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG pipeline...")
        # Access variables from .env file
        self.file_path = os.environ.get('PDF_FILE_PATH')
        self.chroma_db_path = os.environ.get('CHROMA_DB_PATH')
        
        if not all([self.file_path, self.chroma_db_path, os.environ.get("GOOGLE_API_KEY")]):
            raise ValueError("Required environment variables are not set.")

        self.embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.collection_name = "pdf_docs_collection"

        self.vectorstore = self._get_or_create_vectorstore()
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)
        self.qa_chain = self._setup_qa_chain()
        print("RAG pipeline initialized successfully.")

    def _get_or_create_vectorstore(self):
        """Initializes or loads the Chroma vector store."""
        if os.path.exists(self.chroma_db_path) and os.listdir(self.chroma_db_path):
            print("Loading vector store from disk...")
            vectorstore = Chroma(
                persist_directory=self.chroma_db_path,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
        else:
            print("Generating new embeddings and saving to disk...")
            loader = PyPDFLoader(self.file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding_model,
                persist_directory=self.chroma_db_path,
                collection_name=self.collection_name
            )
            print("Embeddings generated and saved.")
        
        print(f"Number of documents in vector store: {vectorstore._collection.count()}")
        return vectorstore

    def _setup_qa_chain(self):
        """Sets up the RetrievalQA chain."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful assistant. Use the following context to answer the question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question:
            {question}

            Helpful Answer:
            """
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt_template}
        )

    def get_response(self, query: str) -> str:
        """Runs the query through the RAG chain and returns the response."""
        result = self.qa_chain({"query": query})
        return result["result"]
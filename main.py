# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
from rag_pipeline import RAGPipeline

# Pydantic model for the request body
class ChatRequest(BaseModel):
    query: str

# Global variable to hold our RAG pipeline instance
rag_app: RAGPipeline = None

# Using an async context manager to handle application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on application startup
    print("Application startup event: Initializing RAG pipeline...")
    try:
        global rag_app
        rag_app = RAGPipeline()
        print("RAG pipeline is ready.")
    except Exception as e:
        print(f"Error during RAG pipeline initialization: {e}")
        # In a real app, you might want to handle this more gracefully
        raise
    
    # This part yields control back to FastAPI to start serving requests
    yield
    
    # This code runs on application shutdown
    print("Application shutdown event: Cleaning up...")
    # Add any cleanup code here if needed
    
# Initialize the FastAPI app with the lifespan context manager
app = FastAPI(
    title="PDF RAG Chatbot API",
    description="A simple API for a Retrieval-Augmented Generation chatbot that answers questions based on a PDF document.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the PDF RAG Chatbot API! Use the /chat endpoint to ask questions."}

@app.post("/chat")
async def chat_with_document(request: ChatRequest):
    """
    Endpoint to get a response from the RAG chatbot.
    """
    if rag_app is None:
        raise HTTPException(status_code=503, detail="RAG pipeline is not initialized yet. Please try again in a moment.")
    
    try:
        response = rag_app.get_response(request.query)
        return {"response": response}
    except Exception as e:
        print(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
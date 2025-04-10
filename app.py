from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import numpy as np
import os
import shutil
import uvicorn
import json
import traceback

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create necessary directories
Path("documents").mkdir(exist_ok=True)
Path("embeddings").mkdir(exist_ok=True)
Path("static").mkdir(exist_ok=True)

# Mount static files with explicit paths
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text):
    """Generate embeddings for a given text"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use mean pooling
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    return embeddings.numpy()[0]

# Store for documents and their embeddings
document_stores = {}

def save_embeddings(filename, texts, embeddings_list):
    """Save document chunks and their embeddings to disk"""
    data = {
        "texts": [{"content": doc.page_content, "metadata": doc.metadata} for doc in texts],
        "embeddings": [emb.tolist() for emb in embeddings_list]
    }
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(data, f)

def load_embeddings(filename):
    """Load document chunks and their embeddings from disk"""
    try:
        with open(f"embeddings/{filename}.json", "r") as f:
            data = json.load(f)
        return data
    except:
        return None

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/style.css")
async def get_style():
    return FileResponse("static/style.css")

@app.get("/script.js")
async def get_script():
    return FileResponse("static/script.js")

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        # Get list of PDF files in documents directory
        pdf_files = [f for f in os.listdir("documents") if f.endswith('.pdf')]
        return pdf_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save the uploaded file
        file_path = f"documents/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        try:
            # Load and split the document
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(pages)
            
            # Generate embeddings for each chunk
            embeddings_list = []
            for text in texts:
                try:
                    embedding = get_embeddings(text.page_content)
                    embeddings_list.append(embedding)
                except Exception as e:
                    print(f"Error generating embedding for chunk: {str(e)}")
                    continue
            
            if not embeddings_list:
                raise Exception("Failed to generate embeddings for any text chunks")
            
            # Save embeddings to disk
            save_embeddings(file.filename, texts, embeddings_list)
            
            # Store in memory
            document_stores[file.filename] = {
                "texts": texts,
                "embeddings": embeddings_list
            }
            
            return {"message": f"Document {file.filename} processed successfully"}
        except Exception as e:
            # Clean up the uploaded file if processing fails
            if os.path.exists(file_path):
                os.remove(file_path)
            print(f"Error processing document: {str(e)}")
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/ask")
async def ask_question(document_name: str = Form(...), question: str = Form(...)):
    print(f"Received question for document: {document_name}")
    print(f"Question: {question}")
    
    if not document_name or not question:
        raise HTTPException(status_code=400, detail="Document name and question are required")
    
    if document_name not in document_stores:
        # Try to load from disk if not in memory
        data = load_embeddings(document_name)
        if data is None:
            raise HTTPException(status_code=404, detail="Document not found or not processed")
        
        document_stores[document_name] = {
            "texts": [text for text in data["texts"]],
            "embeddings": [np.array(emb) for emb in data["embeddings"]]
        }
    
    try:
        # Get question embedding
        question_embedding = get_embeddings(question)
        
        # Calculate similarities
        store = document_stores[document_name]
        similarities = []
        for emb in store["embeddings"]:
            similarity = np.dot(question_embedding, emb) / (np.linalg.norm(question_embedding) * np.linalg.norm(emb))
            similarities.append(similarity)
        
        # Get top 3 most similar chunks
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_texts = [store["texts"][i] for i in top_indices]
        
        # Combine and structure the response
        combined_content = " ".join([doc.page_content for doc in top_texts])
        
        # Create a natural language response
        response = {
            "answer": f"D'après le document, {combined_content}",
            "confidence": float(np.mean([similarities[i] for i in top_indices]))
        }
        
        return response
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
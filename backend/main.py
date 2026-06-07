from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from memory import memory_store

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*",'http://localhost:5173'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
# Assuming Ollama is running locally on default port 11434
embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = ChatOllama(model="llama3.2", temperature=0)

# Initialize ChromaDB
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

class ChatRequest(BaseModel):
    query: str
    session_id: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the file temporarily
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Extract text using PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        # Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        
        # Store in ChromaDB
        vectorstore.add_documents(splits)
        
        return {"message": "Successfully processed and stored PDF."}
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query
    
    # Get chat history
    chat_history = memory_store.get_history_string(session_id)
    
    # 1. Reformulate Query
    reformulate_prompt = ChatPromptTemplate.from_template(
        "Given the following chat history and a new user question, "
        "rephrase the new question to be a standalone question that can be "
        "understood without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is.\n\n"
        "Chat History:\n{chat_history}\n\n"
        "New Question: {question}\n\n"
        "Standalone Question:"
    )
    
    reformulate_chain = reformulate_prompt | llm | StrOutputParser()
    
    if chat_history.strip():
        standalone_query = reformulate_chain.invoke({
            "chat_history": chat_history,
            "question": user_query
        })
    else:
        standalone_query = user_query
        
    # 2. Retrieve Context
    docs = retriever.invoke(standalone_query)
    context = "\n\n".join([doc.page_content for doc in docs])
    sources = [doc.page_content for doc in docs] # just returning the chunk texts for simplicity
    
    # 3. Generate Answer
    qa_prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    answer = qa_chain.invoke({
        "context": context,
        "question": standalone_query
    })
    
    # Update memory
    memory_store.add_message(session_id, "user", user_query)
    memory_store.add_message(session_id, "assistant", answer)
    
    return {
        "answer": answer,
        "sources": sources,
        "standalone_query": standalone_query
    }

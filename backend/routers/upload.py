"""
PDF Upload Router
==================
Handles PDF document upload, parsing, chunking, and vector storage.
Extracted from the original main.py with zero functional changes.
"""

import os
import shutil

from fastapi import APIRouter, UploadFile, File

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from db.collections import vectorstore

router = APIRouter(tags=["Upload"])


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing.
    The file is parsed, chunked, and stored in the ChromaDB vector store.
    """
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

        return {
            "message": "Successfully processed and stored PDF.",
            "chunks_created": len(splits),
            "pages_processed": len(docs),
        }
    finally:
        # Cleanup
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

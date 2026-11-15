import os 
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader, PyPDFLoader)

from dotenv import load_dotenv

load_dotenv()

def load_text_file():
    # Create a temporary text file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"This is a sample text file for testing.")
        temp_file_path = temp_file.name

    # Load the text file using TextLoader
    loader = TextLoader(temp_file_path)
    documents = loader.load()

    for doc in documents:
        print(f"Loaded document: {doc}")
        print(f"Loaded document metadata: {doc.metadata}")
        print(f"Loaded document content: {doc.page_content}")

    # Clean up the temporary file
    os.remove(temp_file_path)

    return documents

def load_pdf_file(pdf_path: str):
    # Load the PDF file using PyPDFLoader
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        print(f"Loaded document: {doc}")
        print(f"Loaded document metadata: {doc.metadata}")
        print(f"Loaded document content: {doc.page_content}")

    return documents

if __name__ == "__main__":
    # load_text_file()
    load_pdf_file(Path(__file__).parent / "docs" / "iso27001.pdf")
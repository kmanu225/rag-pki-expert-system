from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from typing import List
import os


def load_and_split_texts(
    directory: str, chunk_size: int = 1000, chunk_overlap: int = 200
):
    """
    Loads all .txt files from a directory and splits them into passages using LangChain's RecursiveCharacterTextSplitter.

    Args:
        directory (str): Path to the directory containing text files.
        chunk_size (int): Maximum size of each text chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List[str]: List of text chunks/passages.
    """
    passages = []

    if not os.path.isdir(directory):
        raise ValueError(f"Directory not found: {directory}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Only process text files
        if filename.lower().endswith(".txt") and os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                passages.extend(text_splitter.split_text(text))

    return passages


def store_passages_in_chromadb(
    passages: List[str], collection_name: str = "rag_cookbook_collection"
):
    """
    Stores a list of text passages into a ChromaDB collection.

    Args:
        passages (List[str]): List of text chunks to store.
        collection_name (str): Name of the ChromaDB collection.

    Returns:
        chromadb.Collection: The ChromaDB collection containing the passages.
    """
    if not passages:
        raise ValueError("No passages to store.")

    # Initialize ChromaDB client and collection
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Add documents with unique string IDs
    collection.add(documents=passages, ids=[str(i) for i in range(len(passages))])

    return collection
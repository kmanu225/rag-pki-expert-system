from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from typing import List
import os


CHROMA_DB_PATH = "db/chroma_db"


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


def create_local_contextual_knowledge(
    passages: List[str],
    collection_name: str = "knwoledge_base_collection",
    persist_directory: str = CHROMA_DB_PATH,
):
    """
    Stores a list of text passages into a persistent ChromaDB collection.

    Args:
        passages (List[str]): List of text chunks to store.
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where the ChromaDB database will be stored.

    Returns:
        chromadb.Collection: The ChromaDB collection containing the passages.
    """
    if not passages:
        raise ValueError("No passages to store.")

    # Initialize ChromaDB client with persistence
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Add documents with unique string IDs
    collection.add(documents=passages, ids=[str(i) for i in range(len(passages))])


def get_local_contextual_knowledge(
    collection_name: str = "knwoledge_base_collection",
    persist_directory: str = CHROMA_DB_PATH,
):
    """
    Retrieves an existing persistent ChromaDB collection.

    Args:
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where the ChromaDB database is stored.

    Returns:
        chromadb.Collection: The ChromaDB collection.
    """
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name}' not found in {persist_directory}: {e}"
        )
    return collection


def create_remote_contextual_knowledge(
    passages: List[str],
    collection_name: str = "knwoledge_base_collection",
    host: str = "localhost",
    port: int = 8000,
):
    """
    Stores a list of text passages into a persistent ChromaDB collection.

    Args:
        passages (List[str]): List of text chunks to store.
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where the ChromaDB database will be stored.

    Returns:
        chromadb.Collection: The ChromaDB collection containing the passages.
    """
    if not passages:
        raise ValueError("No passages to store.")

    # Initialize ChromaDB client with persistence
    print(f"Connecting to ChromaDB at {host}:{port}...")
    chroma_client = chromadb.HttpClient(host, port)
    
    print(f"Creating or getting collection '{collection_name}'...")
    collection = chroma_client.get_or_create_collection(name=collection_name)

    # Add documents with unique string IDs
    collection.add(documents=passages, ids=[str(i) for i in range(len(passages))])



def get_remote_contextual_knowledge(
    collection_name: str = "knwoledge_base_collection",
    host: str = "localhost",
    port: int = 8000,
):
    """
    Retrieves an existing persistent ChromaDB collection.

    Args:
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory where the ChromaDB database is stored.

    Returns:
        chromadb.Collection: The ChromaDB collection.
    """
    chroma_client = chromadb.HttpClient(host, port)
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(
            f"Collection '{collection_name}' not found on {host}:{port}: {e}"
        )
    return collection



if __name__ == "__main__":
    knowledge_base = "src/knowledge"
    passages = load_and_split_texts(knowledge_base)
    
    # Local storage 
    # create_local_contextual_knowledge(passages)
    # collection = get_local_contextual_knowledge()
    # print(f"Collection metadata: {collection.name}, {collection.count()} documents.")
    
    # Remote storage
    create_remote_contextual_knowledge(passages)
    remote_collection = get_remote_contextual_knowledge()
    print(f"Remote collection metadata: {remote_collection.name}, {remote_collection.count()} documents.")

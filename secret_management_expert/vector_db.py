import chromadb
from typing import List

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


from langchain.text_splitter import RecursiveCharacterTextSplitter
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

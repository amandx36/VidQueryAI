import os
import hashlib
from langchain_community.vectorstores import Chroma


#  Normalize text 
def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


#  Generate unique ID for each chunk using video_id 
def generate_id(text: str, video_id: str) -> str:
    clean_text = normalize_text(text)
    return hashlib.md5((video_id + clean_text).encode()).hexdigest()
# md5 is an algorithm that takes input and generates a 32-character unique string


#  Main function
def create_vector_store(chunks, model, video_id: str, vector_store=None):
    try:
        persist_directory = os.path.abspath("../VectorDataBase/chroma_db")
        os.makedirs(persist_directory, exist_ok=True)  #  ensure folder exists

        db_file = os.path.join(persist_directory, "chroma.sqlite3")

        # Prepare documents + IDs
        docs = []
        ids = []

        for chunk in chunks:
            text = chunk.page_content
            doc_id = generate_id(text, video_id)

            # preserve existing metadata
            chunk.metadata = {
                **(chunk.metadata or {}),
                "video_id": video_id,
                "chunk_id": doc_id
            }

            docs.append(chunk)
            ids.append(doc_id)

        # If vector_store already passed → reuse directly
        if vector_store:
            print("Using existing vector store (in-memory)")
            vector_store.add_documents(documents=docs, ids=ids)
            return vector_store

        # load existing DB
        if os.path.exists(db_file):
            print("Loading existing vector store")

            vector_store = Chroma(
                collection_name="youtube_chunks",
                embedding_function=model,
                persist_directory=persist_directory
            )

            vector_store.add_documents(documents=docs, ids=ids)

        # create new DB
        else:
            print("Creating new vector store")

            vector_store = Chroma.from_documents(
                documents=docs,
                embedding=model,
                ids=ids,
                collection_name="youtube_chunks",
                persist_directory=persist_directory
            )

        return vector_store

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None
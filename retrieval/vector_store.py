#indexing steps using chroma vector store 

# take all the chunks and create a vector store using chroma 


#model is ollma object which is used to generate the embeddings for the chunks

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma



#This function needs object of documents 
def create_vector_store(chunks, model):
    try:
        return Chroma.from_documents(
            documents = chunks,
            embedding= model,
            persist_directory="./chroma_db"
            
            )
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return "Unable to create vector store"


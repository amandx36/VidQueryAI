#this the main file which contains the pipeline 

from ingestion.youtube_loader import  load_youtube_data

from llm.generator import get_llm_model
from llm.prompts import get_prompt
from processing.chunker import chunk_splitter

from retrieval.embedder import get_embeddings_from_ollama
from retrieval.vector_store import create_vector_store
from retrieval.retriever import fetcher
from utils.formatter import format_docs 

from retrieval.embedder import Embeding_model
from langchain_core.documents import Document

# pipeline of chat RAG system

def run_pipeline(video_id : str , user_query : str  ):
    # step 1 load the utube data 
    video_data = load_youtube_data(video_id, language="en")

    # step 2 chunk the data into different chuks 
    chunks = chunk_splitter(video_data)

    #step 3 create the vector store using ollma and chroma and store in chroma db 


    chunkList : list[Document] = []
    for chunk in chunks:
        chunk_embedding = get_embeddings_from_ollama(chunk)
        print(f"Chunk: {chunk}\nEmbedding: {chunk_embedding}\n")
        chunkList.append((chunk, chunk_embedding))

    vector_store = create_vector_store(chunkList, Embeding_model)

    # step 4 retrival of data using the user query and vector db 

    # step 5 formating the content 
    content = format_docs(chunks)
    retrieved_chunks = fetcher(vector_store, user_query)

    # Step 5  prompt generating 
    prompt = get_prompt(retrieved_chunks, user_query)
    
    # step 6 for generating the response using llm 

    llm = get_llm_model()
    response = llm(prompt)

    return response

if __name__ == "__main__":
    video_id = "3T3wMHcrJTuMida5" 
    user_query = "What is the main topic of the video?"
    response = run_pipeline(video_id, user_query)
    print("Response from RAG system:", response)




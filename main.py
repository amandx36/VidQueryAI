#this the main file which contains the pipeline 

from ingestion.youtube_loader import  load_youtube_data

from llm.generator import get_llm_model
from llm.prompts import get_prompt
from processing.chunker import chunk_splitter

from retrieval.embedder import get_embeddings_from_ollama
from retrieval.vector_store import create_vector_store
from retrieval.retriever import fetcher



# pipeline of chat RAG system

def run_pipeline(video_id , user_query ):
    # step 1 load the utube data 
    video_data = load_youtube_data(video_id, language="en")

    # step 2 chunk the data into different chuks 
    chunks = chunk_splitter(video_data)

    #step 3 create the vector store using ollma and chroma 
    
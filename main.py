#this the main file which contains the pipeline 

from ingestion.youtube_loader import  load_youtube_data

from llm.generator import get_llm_model
from llm.prompts import get_prompt
from processing.chunker import chunk_splitter

from retrieval.vector_store import create_vector_store
from retrieval.retriever import fetcher
from utils.formatter import format_docs 

from retrieval.embedder import Embeding_model

# for extracting the youtube  url 

from urllib.parse import urlparse, parse_qs

# pipeline of chat RAG system

def run_pipeline(video_id: str, user_query: str, vector_store=None):
    # step 1 load the utube data (only if vector_store not exists)
    if not vector_store:
        video_data = load_youtube_data(video_id, language="en")

        # step 2 chunk the data into different chunks 
        chunks = chunk_splitter(video_data)

        # step 3 create the vector store
        vector_store = create_vector_store(chunks, Embeding_model, video_id)

    # step 4 retrieval of data using the user query and vector db 
    retrieved_chunks = fetcher(vector_store, user_query)

    # step 5 formatting the content 
    content = format_docs(retrieved_chunks)
    
    # step 6 prompt generating 
    prompt_template = get_prompt()
    prompt = prompt_template.format(retrieved_chunks=content, question=user_query)  
    
    # step 7 generate the response using llm 
    llm = get_llm_model()
    response = llm.invoke(prompt)

    return response, vector_store  
def extract_video_id(url:str)-> str:
    try:
        parsed_url = urlparse(url)

        # Case 1: normal youtube link
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]

        # Case 2: shortened link (youtu.be)
        if parsed_url.hostname == "youtu.be":
            return parsed_url.path.lstrip("/")

        return None

    except Exception:
        return None

if __name__ == "__main__":
    vector_store = None
    video_id = None

    while True:
        print("\n1. Load video")
        print("2. Ask question")
        print("3. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            url = input("Enter YouTube URL: ")
            video_id = extract_video_id(url)

            if not video_id:
                print("Invalid URL")
                continue

            # reset store for new video
            vector_store = None
            print("Video loaded")

        elif choice == "2":
            if not video_id:
                print("Load video first")
                continue

            query = input("Enter your question: ")

            response, vector_store = run_pipeline(
                video_id,
                query,
                vector_store
            )

            print("\nAnswer:", response)

        elif choice == "3":
            break

        else:
            print("Invalid choice")
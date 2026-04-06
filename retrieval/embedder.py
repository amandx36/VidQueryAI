# This function is used to create or generating the embeddings from ollama

from langchain_ollama import OllamaEmbeddings

Embeding_model = OllamaEmbeddings(
    model = "nomic-embed-text:latest"
)

def get_embeddings_from_ollama(text :str):
    try:
        Embeddings = Embeding_model.embed_query(text)
        return Embeddings
    except Exception as e:
        print(f"Error generating embeddings from Ollama: {e}")
        return "Unable to generate embedding fom ollama"

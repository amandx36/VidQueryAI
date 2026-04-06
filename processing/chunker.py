from langchain_text_splitters import RecursiveCharacterTextSplitter

#spliting the text into chunks 

def chunk_splitter (transcript_text : str):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        # print(text_splitter)
        chunks = text_splitter.create_documents([transcript_text])
        # print(chunks)
        return chunks
    except Exception as e:
        print(f"Error splitting transcript text: {e}")
        return []


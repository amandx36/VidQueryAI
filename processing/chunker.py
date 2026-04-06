from langchain_text_splitters import RecursiveCharacterTextSplitter

#spliting the text into chunks 

def chunk_splitter (transcript_text : str):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
        chunks = text_splitter.create_documents([transcript_text])
        return chunks
    except Exception as e:
        print(f"Error splitting transcript text: {e}")
        return ["Not able to split transcript text"] ;

document = chunk_splitter("This is a sample transcript text that needs to be split into smaller chunks for processing. The text splitter will create chunks of a specified size with some overlap to ensure that the context is preserved across chunks.")

print(document)
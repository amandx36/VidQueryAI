from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

# select model

def get_llm_model():
    try :
        llm= GoogleGenerativeAI(model="gemini-2.5-flash-lite",temperature=0.7)
        return llm 
    except Exception as e:
        print(f"Error initializing LLM model: {e}")
        return "Unable to initialize LLM model"

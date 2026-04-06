from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate           



def get_prompt():
    system_message = SystemMessagePromptTemplate.from_template(
        """You are an intelligent assistant specialized in answering questions from a YouTube video transcript.

Instructions:
- Use ONLY the provided context to answer the question.
- Do NOT make up information or hallucinate.
- If the answer is partially available, provide the best possible answer from the context.
- If the answer is not present, clearly say: ["I don't know based on the provided context."]
- Be concise, clear, and informative. {retrieved_chunks}"""
    )
    
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    return prompt
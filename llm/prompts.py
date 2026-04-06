from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate           



def get_prompt():
    system_message = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant for answering questions about the content of the video  if context is insufficient that say i dont  know . You have access to the following information: {retrieved_chunks}"
    )
    
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    prompt = ChatPromptTemplate.from_messages([system_message, human_message])
    
    return prompt
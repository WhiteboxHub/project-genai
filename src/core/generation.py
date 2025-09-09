import os
from langchain_groq import ChatGroq

def generate_response(context: str, question: str):
  
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Please export it or add to your .env file.")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",  
        api_key=api_key,
        temperature=0.2
    )
    
    chat_prompt = create_chat_prompt_template()
    formatted_prompt = chat_prompt.format_prompt(context=context, question=question)
    
    result = llm.invoke(formatted_prompt.to_messages())
    
    return result.content

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
from langchain_groq import ChatGroq
from prompts import create_chat_prompt_template

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
    # Format the prompt with the given context and question
    formatted_prompt = chat_prompt.format_messages(
        context=context,
        question=question
    )

    response = llm.invoke(formatted_prompt)
    return response.content
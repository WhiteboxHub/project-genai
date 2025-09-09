from groq import Groq
import os
from dotenv import load_dotenv
import langchain_groq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Load variables from .env file
load_dotenv()

class generation_ans:
    @staticmethod
    def generate_with_groq(docs:list, query: str, prompt: str, model: str = "llama-3.3-70b-versatile") -> str:
        """
        Use Groq LLM to answer a query based on given docs and prompt.
    
        Parameters:
            docs   : list of strings (documents or text chunks)
            query  : user question
            prompt : instruction / system message
            model  : Groq LLM name (default = llama-3.3-70b-versatile)
    
        Returns:
            str -> model's response
        """
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        # Combine docs into context
        context = "\n\n".join(docs)

        # Build full prompt
        fullprompt = f"{prompt}\n\nContext:\n\n{context}\n\nQuestion:\n{query}"       
        
        # Call Groq LLM
        response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": fullprompt}]
        )       
        return response.choices[0].message.content

    # Example usage
if __name__ == "__main__":
    docs = [
        "The moon orbits the Earth every 27.3 days.",
        "Neil Armstrong was the first person to walk on the moon in 1969."
    ]
    query = "Who was the first person on the moon?"
    prompt = "You are a helpful assistant. Answer clearly."
    answer = generation_ans.generate_with_groq(docs, query, prompt, model="llama-3.3-70b-versatile")

    print("Answer:", answer)

    ####output#############
    '''
    Answer: The first person to walk on the moon was Neil Armstrong, in 1969.
    '''
   


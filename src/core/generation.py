import os
from groq import Groq
from dotenv import load_dotenv


class LLMGenerator:
    def __init__(self, api_key: str):
        """
        Initializes the Groq client.

        Args:
            api_key (str): Your Groq API key.
        """
        self.client = Groq(api_key=api_key)


    def generate_response_from_documents(self, k_docs, query):
        """
        A class to handle text generation using a Groq model,
        integrating a user's query with retrieved documents.
        """
        prompt = (
            "You are a helpful assistant. Use the following documents to answer the user's query. "
            "If you cannot find the answer in the documents, state that you do not have enough information "
            "and do not try to make up an answer."
            f"\n\n### Retrieved Documents:\n{k_docs}\n\n### User Query:\n{query}"
        )
        try:
            completion = self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=2048,
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"An error occurred during API call: {e}"


if __name__ == "__main__":
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    obj = LLMGenerator(groq_api_key)


        

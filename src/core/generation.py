import os
from groq import Groq
from src.utils.logger import logger
from typing import List, Optional

class LLMGeneration:
    def __init__(self):
        # Initialize Groq client with API key from environment
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Default to mixtral-8x7b model
        self.model = "mixtral-8x7b-v0.1"

    @logger
    def groqans(self, k_doc: List[str], query: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate an answer using Groq LLM based on retrieved documents and query.
        
        Args:
            k_doc: List of relevant document chunks
            query: User's query
            system_prompt: Optional system prompt to guide the model's behavior
            
        Returns:
            str: Generated answer from the LLM
        """
        # Combine documents into context
        context = "\n".join(k_doc)
        
        # Prepare messages
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add context and query
        messages.append({
            "role": "user",
            "content": f"""Given the following context:
            
{context}

Answer this question: {query}

Please provide a clear and concise answer based only on the given context. If the answer cannot be found in the context, say so."""
        })

        try:
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=1000,
                top_p=1,
                stream=False
            )
            
            # Extract and return the response
            return completion.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response with Groq: {str(e)}")
            raise
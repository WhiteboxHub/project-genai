
import os
from langchain_groq import ChatGroq
from prompt import format_prompt
from dotenv import load_dotenv
load_dotenv()


class GeneralLLM:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

    def gen_ans(self, retrieved_chunks_docs, query: str, system_prompt: str = None):
        """
        Generate an answer from Groq LLM based on retrieved docs and query.
        Supports list of strings, dicts, or LangChain Documents.
        """
        if not retrieved_chunks_docs:
            raise ValueError("retrieved_chunks_docs is empty")

        # case 1: list of dicts (your MilvusDB with metadata)
        if isinstance(retrieved_chunks_docs[0], dict):
            context = "\n".join([doc["text"] for doc in retrieved_chunks_docs])

        # case 2: list of LangChain Documents
        elif hasattr(retrieved_chunks_docs[0], "page_content"):
            context = "\n".join([doc.page_content for doc in retrieved_chunks_docs])

        # case 3: plain list of strings
        elif isinstance(retrieved_chunks_docs[0], str):
            context = "\n".join(retrieved_chunks_docs)

        else:
            raise TypeError(
                f"Unsupported type for retrieved_chunks_docs: {type(retrieved_chunks_docs[0])}"
            )

        # Build final prompt
        final_prompt = format_prompt(query, context)
        response = self.llm.invoke(final_prompt)

        return response.content

from storing import MilvusDB
from embeding import EmbedModel
from general import GeneralLLM
from prompt import format_prompt   # import formatter

def query_pipeline(query: str, k: int = 3):
    # init db + llm
    db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())
    llm = GeneralLLM()

    # retrieve context
    #results = db.retrieve_data(query, k=k)
    results = db.search(query, k=k)
    if not results:
        print("⚠️ No relevant docs found in Milvus.")
        return

    context = "\n".join([r["text"] for r in results])

    # build full prompt for LLM
    full_prompt = format_prompt(query, context)

    # ask Groq
    answer = llm.gen_ans(full_prompt, query)

    print("\n🤖 Groq Answer:")
    print(answer)

if __name__ == "__main__":
    queries = [
        "What is Agentic AI?",
        "How does chunking help embeddings?",
        "Explain what LangChain is"
    ]

    for q in queries:
        print(f"\n=== Query: {q} ===")
        query_pipeline(q)

#==================OUTPUT===========================
# 🤖 Groq Answer:
# Agentic AI refers to a type of artificial intelligence that is designed to take on a more proactive and autonomous role, acting as an agent to achieve specific goals or objectives. This type of AI is programmed to make decisions, take actions, and interact with its environment in a more independent and dynamic way.

# === Query: How does chunking help embeddings? ===
# ✅ Connected to Milvus at localhost:19530

# 🤖 Groq Answer:
# Chunking helps embeddings by breaking down complex information into smaller, manageable pieces. 

# This process enables the AI to:

# 1. Reduce the dimensionality of the data, making it easier to process and analyze.
# 2. Identify patterns and relationships within the data.
# 3. Improve the accuracy and efficiency of the embedding generation process.
# 4. Allow for more nuanced and detailed representations of the data.

# By chunking, the AI can create more informative and contextually relevant embeddings, leading to better performance in downstream tasks and applications.

# === Query: Explain what LangChain is ===
# ✅ Connected to Milvus at localhost:19530

# 🤖 Groq Answer:
# LangChain is a Python library designed to facilitate the construction of conversational
# AI systems. It utilizes various techniques such as chaining, threading, and looping to enable the creation of complex conversational flows and interfaces. LangChain aims to simplify the development of AI assistants, chatbots, or other conversational agents by providing a flexible and modular framework for building and combining different components. 
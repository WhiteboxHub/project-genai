# from prompt import format_prompt
# from storing import MilvusDB
# from embeding import EmbedModel

# # Initialize Milvus
# db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())

# # Your query
# query = "What is in the document?"

# # Retrieve relevant context from Milvus
# retrieved_docs = db.retrieve_data(query, k=3)
# context = "\n".join([doc["text"] for doc in retrieved_docs])

# # Format prompt
# prompt = format_prompt(query, context)
# print(prompt)

#==================output====================
# System: You are an AI assistant. 
# Use only the provided context to answer questions.
# Be clear and concise.

# Context:
# Slide 2: Learning Objectives
# Explain what students will learn: 1.
# Slide 3: What is a Decision Tree?
# Emphasize that this session will cover both conceptual understanding and
# practical implementation in Python.

# User Question: What is in the document?

from storing import MilvusDB
from embeding import EmbedModel
from general import GeneralLLM
from prompt import format_prompt   # import formatter

def query_pipeline(query: str, k: int = 3):
    # init db + llm
    db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())
    llm = GeneralLLM()

    # retrieve context
    results = db.retrieve_data(query, k=k)
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
    query_pipeline("Explain decision trees vs random forests")

#========output============
# 🤖 Groq Answer:
# Decision Trees and Random Forests are both machine learning algorithms used for classification and regression tasks.

# **Decision Trees:**
# A decision tree is a tree-like model where each internal node represents a feature or attribute, each branch represents a decision, and each leaf node represents a class label or target value. The tree is built by recursively partitioning the data into subsets based on the most significant feature.

# **Random Forests:**
# A Random Forest is an ensemble learning method that combines multiple decision trees to improve the accuracy and robustness of predictions. Instead of using the entire feature set to build each tree, a random subset of features is selected, reducing overfitting and improving generalization.
    
# Key differences:

# - **Ensemble method**: Random Forest is an ensemble method that combines multiple decision trees, while Decision Trees are single models.
# - **Feature selection**: Random Forest selects a random subset of features at each node, whereas Decision Trees use all features.
# - **Robustness**: Random Forests are more robust to overfitting due to the random feature selection, whereas Decision Trees can overfit the data.

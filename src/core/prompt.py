

def system_prompt():
    return """You are an AI assistant. 
Use only the provided context to answer questions.
Be clear and concise."""

def format_prompt(query: str, context: str):
    return f"""
System: {system_prompt()}

Context:
{context}

User Question: {query}
"""

from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

def create_chat_prompt_template():
    system_template = (
        "You are a helpful assistant. Use the following context to answer the user's question.\n"
        "If the answer is not in the context, say \"I don't know based on the provided information.\""
        "\n\nContext:\n{context}"
    )
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt
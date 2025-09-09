from storing import MilvusDB
from embeding import EmbedModel
from general import GeneralLLM
from prompt import format_prompt

def query_pipeline(query: str, k: int = 3):
    db = MilvusDB(embedding_model=EmbedModel.huggingface_embedding())
    llm = GeneralLLM()

    # Run both similarity searches
    results_cosine = db.ann_cosine_search(query, k=k)
    results_ip = db.ip_search(query, k=k)

    if not results_cosine and not results_ip:
        print("⚠️ No relevant docs found in Milvus.")
        return

    # Display results
    print("\n--- Cosine Similarity Results ---")
    for r in results_cosine:
        print(r)

    print("\n--- Inner Product Results ---")
    for r in results_ip:
        print(r)

    # Merge results for LLM input (remove duplicates by text)
    combined_results = {r['text']: r for r in results_cosine + results_ip}
    merged_results = list(combined_results.values())

    # Build full prompt for Groq LLM
    full_prompt = format_prompt(query, "\n".join([r["text"] for r in merged_results]))
    answer = llm.gen_ans(merged_results, query)

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
#✅ Connected to Milvus at localhost:19530

# --- Cosine Similarity Results ---
# {'filename': 'agentic-ai.pdf', 'text': 'What is Agentic AI?', 'score': 1.0000001192092896}
# {'filename': 'agentic-ai.pdf', 'text': 'What is agentic AI?', 'score': 1.0000001192092896}
# {'filename': 'agentic-ai.pdf', 'text': 'Agentic AI generally refers to AI systems that \npossess the capacity to make autonomous \ndecisions and take actions to achieve speciﬁc \ngoals with limited or no direct human \nintervention3.', 'score': 0.8518047332763672}

# --- Inner Product Results ---
# {'filename': 'agentic-ai.pdf', 'text': 'What is Agentic AI?', 'score': 1.000000238418579}
# {'filename': 'agentic-ai.pdf', 'text': 'What is agentic AI?', 'score': 1.000000238418579}
# {'filename': 'agentic-ai.pdf', 'text': 'Agentic AI generally refers to AI systems that \npossess the capacity to make autonomous \ndecisions and take action
# 🤖 Groq Answer:
# Agentic AI refers to AI systems that can make autonomous decisions and take actions to achieve specific goals with limited or no direct human intervention.

# === Query: How does chunking help embeddings? ===
# ✅ Connected to Milvus at localhost:19530

# --- Cosine Similarity Results ---
# {'filename': 'chunking_strategies.pdf', 'text': 'Second, chunking can be seen as a more \nautomatic and continuous process that occurs during  perception.', 'score': 0.6079119443893433}
# {'filename': 'chunking_strategies.pdf', 'text': 'The user has r equest ed enhanc ement of the do wnlo aded file.CHUNKING MECHANISMS AND LEARNING  \nFernand Gobet \nDepartment of Psychology, Brunel University \nUxbridge \nUnited Kingdom \nfernand.gobet@brunel.ac.uk \n \nPeter C. R. Lane \nSchool of Computer Science, University of Hertfords hire \nHatfield \nUnited Kingdom \np.c.lane@herts.ac.uk \n \nSynonyms \n \nDefinition \nA chunk is meaningful unit of information built from smalle r pieces of information, and chunking is the \nprocess of creating a new chunk.', 'score': 0.5867968201637268}
# {'filename': 'chunking_strategies.pdf', 'text': 'Here, we talk about perceptual chunking.', 'score': 0.5725541710853577}

# --- Inner Product Results ---
# {'filename': 'chunking_strategies.pdf', 'text': 'Second, chunking can be seen as a more \nautomatic and continuous process that occurs during  perception.', 'score': 0.6079119443893433}
# {'filename': 'chunking_strategies.pdf', 'text': 'The user has r equest ed enhanc ement of the do wnlo aded file.CHUNKING MECHANISMS AND LEARNING  \nFernand Gobet \nDepartment of Psychology, Brunel University \nUxbridge \nUnited Kingdom \nfernand.gobet@brunel.ac.uk \n \nPeter C. R. Lane \nSchool of Computer Science, University of Hertfords hire \nHatfield \nUnited Kingdom \np.c.lane@herts.ac.uk \n \nSynonyms \n \nDefinition \nA chunk is meaningful unit of information built from smalle r pieces of information, and chunking is the \nprocess of creating a new chunk.', 'score': 0.5867968201637268}
# {'filename': 'chunking_strategies.pdf', 'text': 'Here, we talk about perceptual chunking.', 'score': 0.5725541710853577}

# 🤖 Groq Answer:
# Chunking helps embeddings by breaking down complex information into smaller, meaningful units (chunks) that can be easily processed and understood. This process simplifies the information, making it more interpretable and useful for machine learning models, including those used in natural language processing and computer vision.

# === Query: Explain what LangChain is ===
# ✅ Connected to Milvus at localhost:19530

# --- Cosine Similarity Results ---
# {'filename': 'langchain.pdf', 'text': 'LangChain also allows users to save queries, create bookmarks, \nand annotate important sections, enabling efficient retrieval of \nrelevant information from PDF documents.', 'score': 0.5714693069458008}
# {'filename': 'langchain.pdf', 'text': 'LangChain is a cutting -edge solutio n which helps us in the \nquerying proce ss and extracting information from PDFs.', 'score': 0.5709579586982727}
# {'filename': 'langchain.pdf', 'text': 'The features of \nLangChain increase overal l efficiency and makes PDF querying \nmuch easier and  simpler .', 'score': 0.5073256492614746}

# --- Inner Product Results ---
# {'filename': 'langchain.pdf', 'text': 'LangChain also allows users to save queries, create bookmarks, \nand annotate important sections, enabling efficient retrieval of \nrelevant information from PDF documents.', 'score': 0.571469247341156}
# {'filename': 'langchain.pdf', 'text': 'LangChain is a cutting -edge solutio n which helps us in the \nquerying proce ss and extracting information from PDFs.', 'score': 0.5709578990936279}
# {'filename': 'langchain.pdf', 'text': 'The features of \nLangChain increase overal l efficiency and makes PDF querying \nmuch easier and  simpler .', 'score': 0.5073255896568298}

# 🤖 Groq Answer:
# LangChain is a cutting-edge solution that helps with querying and extracting information from PDF documents. It enhances efficiency and simplifies the process of querying PDFs.

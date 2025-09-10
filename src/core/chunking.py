# # handles chunking of long text into smaller pieces that can fit into the model’s max token length (512 tokens for T5).
# chunking.py
# Load model directly
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Model name
# MODEL_NAME = "t5-base"
MODEL_NAME = "google/t5gemma-b-b-prefixlm"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Create pipeline (for summarization instead of text-generation)
# Use a pipeline as a high-level helper
# pipe = pipeline("text-generation", model=MODEL_NAME)
pipe = pipeline("summarization", model=MODEL_NAME)

def chunk_text(text, max_tokens=400, overlap=50):
    """
    Splits long text into smaller overlapping chunks that fit into the model's max length.

    Args:
        text (str): The input text to be chunked.
        max_tokens (int): Maximum tokens per chunk.
        overlap (int): Number of overlapping tokens between chunks.

    Returns:
        list of str: Text chunks.
    """
    tokens = tokenizer.encode(text, return_tensors=None)
    chunks = []

    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks


def summarize_long_text(text):
    """
    Summarize long text by splitting into chunks and summarizing each one.
    """
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        result = pipe(chunk, max_length=150, min_length=30, truncation=True)
        summaries.append(result[0]['summary_text'])

    return " ".join(summaries)


if __name__ == "__main__":
    long_text = """Your very long document goes here..."""
    print("Original length:", len(long_text.split()))
    final_summary = summarize_long_text(long_text)
    print("\nFinal Summary:\n", final_summary)













# from transformers import T5Tokenizer, T5ForConditionalGeneration

# # Load T5-base model and tokenizer
# model_name = "t5-base"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# # Function to chunk text into manageable parts
# def chunk_text(text, max_token_length=512):
#     inputs = tokenizer(text, return_tensors="pt", truncation=False)
#     input_ids = inputs['input_ids'][0]

#     chunks = []
#     for i in range(0, len(input_ids), max_token_length):
#         chunk = input_ids[i:i+max_token_length]
#         chunks.append(chunk)

#     return chunks

# # Function to summarize each chunk
# def summarize_chunks(chunks):
#     summaries = []

#     for chunk in chunks:
#         input_ids = chunk.unsqueeze(0)
#         summary_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         summaries.append(summary)

#     return summaries

# # Main function
# def summarize_large_text(text):
#     print("Original text length:", len(text))
    
#     # Add prefix for summarization
#     text = "summarize: " + text

#     # Chunk and summarize
#     chunks = chunk_text(text)
#     print(f"Text split into {len(chunks)} chunk(s).")
    
#     summaries = summarize_chunks(chunks)
#     final_summary = " ".join(summaries)
    
#     return final_summary

# # Sample usage
# if __name__ == "__main__":
#     long_text = """
#     The T5 model is a transformer-based architecture developed by Google.
#     It is trained on a large corpus of data using a text-to-text framework.
#     T5 can perform a wide range of NLP tasks such as translation, summarization,
#     question answering, and classification by simply changing the task prefix.
#     """

#     summary = summarize_large_text(long_text)
#     print("\nFinal Summary:\n", summary)

# handles chunking of long text into smaller pieces that can fit into the model’s max token length (512 tokens for T5).
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5-base model and tokenizer
model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to chunk text into manageable parts
def chunk_text(text, max_token_length=512):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = inputs['input_ids'][0]

    chunks = []
    for i in range(0, len(input_ids), max_token_length):
        chunk = input_ids[i:i+max_token_length]
        chunks.append(chunk)

    return chunks

# Function to summarize each chunk
def summarize_chunks(chunks):
    summaries = []

    for chunk in chunks:
        input_ids = chunk.unsqueeze(0)
        summary_ids = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return summaries

# Main function
def summarize_large_text(text):
    print("Original text length:", len(text))
    
    # Add prefix for summarization
    text = "summarize: " + text

    # Chunk and summarize
    chunks = chunk_text(text)
    print(f"Text split into {len(chunks)} chunk(s).")
    
    summaries = summarize_chunks(chunks)
    final_summary = " ".join(summaries)
    
    return final_summary

# Sample usage
if __name__ == "__main__":
    long_text = """
    The T5 model is a transformer-based architecture developed by Google.
    It is trained on a large corpus of data using a text-to-text framework.
    T5 can perform a wide range of NLP tasks such as translation, summarization,
    question answering, and classification by simply changing the task prefix.
    """

    summary = summarize_large_text(long_text)
    print("\nFinal Summary:\n", summary)

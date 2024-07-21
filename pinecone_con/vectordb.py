import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# Initialize Pinecone environment
with open('secret.json') as f:
    data = json.load(f)
pinecone_key = data["pinecone_api"]
pc = Pinecone(api_key=pinecone_key)

# Check if the index exists, and if not, create a new one
index_name = 'python-18k-instructions'
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=384, metric='cosine', #dimension 384 for all-miniLM-L12-v2
            spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) )

# Connect to your Pinecone index
index = pc.Index(index_name)

# Load python-code-18k-alpha
df = pd.read_parquet("hf://datasets/iamtarun/python_code_instructions_18k_alpaca/data/train-00000-of-00001-8b6e212f3e1ece96.parquet")

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')  # Example model, replace with your choice

# Generate embeddings
df['embeddings'] = df['instruction'].apply(lambda x: model.encode(x).tolist())

# Prepare the data for upload]
data_to_upload = list(zip(df.index, df['embeddings'], df['output']))


# Function to divide data into chunks for batch processing
def chunked_data(data, chunk_size):
    """Yield successive chunk_size chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Upload data in batches
batch_size = 100
for batch in chunked_data(data_to_upload, batch_size):
    batch_to_upsert = [(str(id), vec, {'output': output}) for id, vec, output in batch]
    index.upsert(vectors=batch_to_upsert)

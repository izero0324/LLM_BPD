import os
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from code_Net_read import python_files_to_df
# Initialize Pinecone environment
with open('secret.json') as f:
    data = json.load(f)
pinecone_key = data["pinecone_api"]
pc = Pinecone(api_key=pinecone_key)

# Check if the index exists, and if not, create a new one
index_name = 'codenet-python800'
if index_name not in pc.list_indexes().names():
    pc.create_index(index_name, dimension=384, metric='cosine', #dimension 384 for all-miniLM-L12-v2
            spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) )

# Connect to your Pinecone index
index = pc.Index(index_name)

# Load codeNet_python800
df = python_files_to_df()
print('data_prepared!')
print(df.head())

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')  # Example model, replace with your choice

code_texts = df['codes'].tolist()


# Generate embeddings
embeddings = model.encode(code_texts, show_progress_bar=True).tolist()

df['embeddings'] = embeddings
# Prepare the data for upload
data_to_upload = list(zip(df.index, df['embeddings'], df['codes']))

# Function to divide data into chunks for batch processing
def chunked_data(data, chunk_size):
    """Yield successive chunk_size chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

# Upload data in batches
batch_size = 100
for batch in chunked_data(data_to_upload, batch_size):
    batch_to_upsert = [(str(id), vec, {'code': codes}) for id, vec, codes in batch]
    try:
        index.upsert(vectors=batch_to_upsert)
    except Exception as e:
        print('error, ', e)
        pass
        
print("Upload complete.")

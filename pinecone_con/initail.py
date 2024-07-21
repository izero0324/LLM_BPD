import os
import json
from pinecone import Pinecone

def pc_connect(index_name = 'python-18k-instructions'):
    # Initialize Pinecone environment
    with open('secret.json') as f:
        data = json.load(f)
    pinecone_key = data["pinecone_api"]
    pc = Pinecone(api_key=pinecone_key)
    if index_name not in pc.list_indexes().names():
        pc.create_index(index_name, dimension=384, metric='cosine', #dimension 384 for all-miniLM-L12-v2
                spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) )
    # Check if the index exists, and if not, create a new one
    # Connect to your Pinecone index
    index = pc.Index(index_name)
    return pc, index
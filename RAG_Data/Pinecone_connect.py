import os
import json
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

def init_pinecone():
    with open('secret.json') as f:
        data = json.load(f)
    pc_api = data['pinecone_api']
    print('API retrieved')
    pc_server = Pinecone(api_key=pc_api)
    print('Pinecone Initiallized')
    return pc_server

def setup_index(index_name, pc_server):

    if index_name not in pc_server.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=2,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        )

def upsert2index(index_name, data, pc_server):
    index = pc_server.Index(index_name)

    index.upsert(
        vectors=[
            {"id": "vec1", "values": [1.0, 1.5]},
            {"id": "vec2", "values": [2.0, 1.0]},
            {"id": "vec3", "values": [0.1, 3.0]},
        ],
        namespace="ns1"
    )

pc = init_pinecone()
setup_index('test-index', pc)
model_name = 'text-embedding-ada-002'

import json
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
# Load api key from secret
with open('secret.json') as f:
    data = json.load(f)
api_key = data["key"]
os.environ["OPENAI_API_KEY"] = api_key # Export the API key

Settings.embed_model = OpenAIEmbedding( model = 'text-embedding-3-small')

from langchain.vectorstores import Pinecone

text_field = "text"
index_name = 'langchain-retrieval-augmentation-fast'
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, Settings.embed_model.embed_query, text_field
)


df = pd.read_parquet("hf://datasets/iamtarun/python_code_instructions_18k_alpaca/\
                     data/train-00000-of-00001-8b6e212f3e1ece96.parquet")
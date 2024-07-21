import json
from sentence_transformers import SentenceTransformer
from pinecone_con.initail import pc_connect



def retrive_docs(input_code, k=3):
    _, index = pc_connect('python-18k-instructions')
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    input_embedding = embedding_model.encode(input_code).tolist()
    retrieved_snippets = index.query(vector=[input_embedding], top_k=k, include_metadata=True)
    retireval_prompt ="\n --- \n  Related optimisations: " + " ".join(
        [snippet['metadata']['output'] for snippet in retrieved_snippets['matches']])
    
    return retireval_prompt
 

 
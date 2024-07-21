import json
from sentence_transformers import CrossEncoder, SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tools.extract import get_data


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
# Load the model, here we use our base sized model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

input = "def is_Power_Of_Two (x): \r\n    return x and (not(x & (x - 1))) \r\ndef differ_At_One_Bit_Pos(a,b): \r\n    return is_Power_Of_Two(a ^ b)"


# Example query and documents
query = "You are a expert Python programmer, and here is your task:\n 1. I will give you a python code and you will rewrite to make it execute faster.\n 2. Do not change the function name!  \n 3. Only return the code.\n 4. Do not output ```python and ```\n 5. Be sure the return is inside the function\n --- \nHere's the code to be optimized:"
# documents = [
#     "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
#     "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
#     "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
#     "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
#     "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
#     "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
# ]

query+=input
query_embedding = model.encode(input).tolist()


# Query the Pinecone index
documents = index.query(vector=[query_embedding], top_k=3, include_metadata=True)

# Fetch the results
for match in documents['matches']:
    print(f"ID: {match['id']}, Score: {match['score']}")


print(documents)
model = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")
# Lets get the scores
#results = model.rank(query, documents, return_documents=True, top_k=3)
# results = model.generate(query)

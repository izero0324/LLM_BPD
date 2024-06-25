import os
import json
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.settings import Settings

# Parameters
LLM_model = "gpt-3.5-turbo"
LLM_model_temp = 0.1
data_dir = "./data/source_files" # path for the RAG data
chunk_size = 1024
index_name = "MyExternalContext" # Index name for vector store

# Load api key from secret
with open('secret.json') as f:
    data = json.load(f)
api_key = data["key"]
os.environ["OPENAI_API_KEY"] = api_key # Export the API key

# Setup LLM connection
Settings.llm = OpenAI(model=LLM_model, temperature=LLM_model_temp)
Settings.embed_model = OpenAIEmbedding()

'''
# Load dataset
documents = SimpleDirectoryReader(input_dir=data_dir).load_data()
#node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# The target key defaults to `window` to match the node_parser's default
postproc = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)


# Define reranker model
rerank = SentenceTransformerRerank(
    top_n = 6, 
    model = "BAAI/bge-reranker-base"
)
'''

def use_RAG(RAG_dataset, code):
    text_list = RAG_dataset
    documents = [Document(text=t['code']) for t in text_list]

    # build index
    index = VectorStoreIndex.from_documents(documents)


    # The QueryEngine class is equipped with the generator
    # and facilitates the retrieval and generation steps
    query_engine = index.as_query_engine()

    # Run your naive RAG query
    response = query_engine.query(
        f"You are a expert Python programmer,rewrite the code make it execute faster. Do not change the function name!  Only return the code. Do not output ```python and ```. Be sure the return is inside th function. \n code to be optimized: {code}"
    )

    #print(response)
    return response
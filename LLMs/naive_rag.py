import os
import json
import weaviate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SimpleNodeParser, SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# Parameters
LLM_model = "gpt-3.5-turbo"
LLM_model_temp = 0.1
data_dir = "./data/source_files" # path for the RAG data
chunk_size = 1024
index_name = "MyExternalContext" # Index name for vector store

# Load api key from secret
print("Load openai key")
with open('secret_openai.json') as f:
    data = json.load(f)
api_key = data["key"]
os.environ["OPENAI_API_KEY"] = api_key # Export the API key

# Setup LLM connection
Settings.llm = OpenAI(model=LLM_model, temperature=LLM_model_temp)
print("LLM model loded")
Settings.embed_model = OpenAIEmbedding()
print("Embedding model loded")

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

# Extract nodes from documents
nodes = node_parser.get_nodes_from_documents(documents)

# Connect to your Weaviate instance
client = weaviate.connect_to_embedded(
    headers={
        "X-OpenAI-Api-Key": api_key  # Replace with your API key
    },
)

# Construct vector store
vector_store = WeaviateVectorStore(
    weaviate_client = client, 
    index_name = index_name
)

# Set up the storage for the embeddings
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Setup the index
# build VectorStoreIndex that takes care of chunking documents
# and encoding chunks to embeddings for future retrieval
index = VectorStoreIndex(
    nodes,
    storage_context = storage_context,
)


# The QueryEngine class is equipped with the generator
# and facilitates the retrieval and generation steps
query_engine = index.as_query_engine(
    node_preprocessors = [postproc],
    vector_store_query_mode="hybrid", 
    alpha=0.5,
    similarity_top_k = 3,
	node_postprocessors = [rerank],
)

# Run your naive RAG query
response = query_engine.query(
    "What happened at Interleaf?"
)

print(response)
print("Test ENDED")
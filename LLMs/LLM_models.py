import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import HuggingFaceHub

# Read API key from the secret file
with open('secret.json') as f:
    data = json.load(f)
api_key = data["key"]
langchain_key = data["langchian_api"]
HuggingFaceHub_key = data["huggingface_api"]

# Export the API key
os.environ["OPENAI_API_KEY"] = api_key

def openAI(model_name, temp=0):
    llm = ChatOpenAI(
                model=model_name,  # insert Change model option!!!
                temperature=temp,
                max_tokens=None,
                timeout=None,
                max_retries=2
            )
    return llm


def llamas(model_name, temp = 0):

    def sellect_llama():
        if model_name == 'llama3':
            repo_id = "meta-llama/Meta-Llama-3-8B"
        else:
            repo_id = "codellama/CodeLlama-7b-hf"
        return repo_id

    repo_id = sellect_llama()
    huggingfacehub_api_token = HuggingFaceHub_key
    llm = HuggingFaceHub(huggingfacehub_api_token = huggingfacehub_api_token,
                        repo_id = repo_id,
                        model_kwargs = {"temperateure":0, "max_new_tokens":500})

    return llm 
    
def mixtral_model(temp=0):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    huggingfacehub_api_token = HuggingFaceHub_key
    llm = HuggingFaceHub(huggingfacehub_api_token = huggingfacehub_api_token,
                        repo_id = repo_id,
                        model_kwargs = {"temperateure":0, "max_new_tokens":500})
    return llm
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
        elif model_name == 'codellama-13b':
            repo_id = "codellama/CodeLlama-13b-hf"
        else:
            repo_id = "codellama/CodeLlama-7b-hf"
        return repo_id

    repo_id = sellect_llama()
    huggingfacehub_api_token = HuggingFaceHub_key
    llm = HuggingFaceHub(huggingfacehub_api_token = huggingfacehub_api_token,
                        repo_id = repo_id,
                        model_kwargs = {"temperateure":0, "max_new_tokens":1024})

    return llm 
    
def mixtral_model(temp=0):
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    huggingfacehub_api_token = HuggingFaceHub_key
    llm = HuggingFaceHub(huggingfacehub_api_token = huggingfacehub_api_token,
                        repo_id = repo_id,
                        model_kwargs = {"temperateure":0, "max_new_tokens":1024})
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    return llm


from transformers import AutoModelForCausalLM, AutoTokenizer

def local_model(temp = 0):
    # Path to your locally saved model
    local_model_path = "models/Meta-Llama-3-8B"
    # Load the tokenizer and model from a local path
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForCausalLM.from_pretrained(local_model_path)

    # You can set any specific model arguments locally as well
    model_kwargs = {
        "temperature": 0,  # Note: corrected spelling from "temperateure" to "temperature"
        "max_new_tokens": 1024
    }

    # # Example on how to use:
    # inputs = tokenizer("Your input text here", return_tensors="pt")
    # outputs = model(**inputs, **model_kwargs)
    # generated_text = tokenizer.decode(outputs[0])
    return model
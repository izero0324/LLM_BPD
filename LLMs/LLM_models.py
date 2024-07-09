import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Read API key from the secret file
with open('secret.json') as f:
    data = json.load(f)
api_key = data["key"]
langchain_key = data["langchian_api"]

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
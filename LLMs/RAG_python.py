import os
import json
import pandas as pd

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.document import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import Language

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA




df = pd.read_parquet("hf://datasets/iamtarun/python_code_instructions_18k_alpaca/data/train-00000-of-00001-8b6e212f3e1ece96.parquet")



text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,chunk_size=2000, chunk_overlap=200
)
texts = text_splitter.create_documents(df['output'].to_list())

with open('secret.json') as f:
    api_key = json.load(f)
os.environ["OPENAI_API_KEY"] = api_key["key"]
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

code_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    verbose=False,
    )
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever(
    search_type="similarity",  # Also test "similarity", "mmr"
    search_kwargs={"k": 5},)

# RAG template
prompt_RAG = """
    You are a proficient python developer. Respond with more efficient code for the original code. Make sure you follow these rules:
    1. Understand the code and how to use it & apply.
    2. Be sure the function name didn't change.
    3. Only return the code.
    4. Do not output```python and ```
    5. check if all the libraries are imported

    Consider these example codes:
    {context}
    """

prompt_RAG_tempate = PromptTemplate(
    template=prompt_RAG, input_variables=['context']
)

qa_chain = RetrievalQA.from_llm(
    llm=code_llm, prompt=prompt_RAG_tempate, retriever=retriever, return_source_documents=True
)

#results = qa_chain({"query": "print('Hello ')\n print('World')"})
#print(results["result"])

def optimize_code_FAISS(code):
    try:
        input = code
        response = qa_chain({"query": input})
        print('=========code==========')
        print(response['result'])
        print('=========end==========')
        return response['result']
    except Exception as e:
        print("An error occurred:", str(e))
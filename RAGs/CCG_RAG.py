
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import LLMs.LLM_models as BaseModels
from tools.retrieval import retrive_docs

with open('secret.json') as f:
    api_key = json.load(f)
os.environ["OPENAI_API_KEY"] = api_key["key"]


# Set code to text model
llm = BaseModels.openAI('gpt-3.5-turbo')
#llm = BaseModels.mixtral_model()
#llm= BaseModels.llamas('llama3')
prompt_template_c2t = ChatPromptTemplate.from_messages([
    ("system", "You are a expert Python programmer, describe the code to make sure a software engineer can reproduce the code by the description. \n Make sure to mention: \n 1. Function name\n 2. The purpose of the code\n3. The input/output format of the code"),
    ("user", "{input}"),
    ("user", "Please Start describing the code, do not repeat the code:")
])
output_parser_c2t = StrOutputParser()
pipeline_c2t = prompt_template_c2t | llm | output_parser_c2t


# Set text to code model
llm2 = BaseModels.llamas('codellama-13b')
#llm2 = BaseModels.mixtral_model()
#llm2 = BaseModels.openAI('gpt-3.5-turbo')
prompt_template_t2c = ChatPromptTemplate.from_messages([
    ("system", "You are a expert Python programmer, write a Python code that fits the description I give you.\
      Make it most efficient and minimise memory usage. \
      Make sure you only return the code and include all the imports.\
      Do not output ```python and ```"),
    ("user", "{input}"),
    ("user", "Please give me the codes only")
])
output_parser_t2c = StrOutputParser()
pipeline_t2c = prompt_template_t2c | llm2 | output_parser_t2c

def interpret_and_describe_code(code):
    # Using OpenAI's GPT model to interpret code and generate a description
    input = code
    response = pipeline_c2t.invoke({"input": input})
    try:
        response = response.split(", do not repeat the code:")[1]
    except:
        response = response
    return response

def generate_code_from_description(description, Retrieval):
    # Generating code from the description using an OpenAI model
    input = description
    if Retrieval:
        input += retrive_docs(input, k=1)
    response = pipeline_t2c.invoke({"input": input})
    try:
        code_response = response.split("Please give me the codes only")[1]
    except:
        code_response = response
    try:
        code_response = code_response.split("System:")[1].split("Human:")[0]
    except:
        code_response = code_response
    try:
        final_response = code_response.split("```python")[1].split("```")[0]
    except:
        final_response = code_response
    #final_response = code_response.split("```python")[1].split("```")[0]
    return final_response

def format_check(result):
    result = result.replace("```python",'')
    result = result.replace("```",'')
    return result

def CCG_RAG(code, debug = False, Retrieval = False):
    try:
        #Step 1: Generate description from code
        code_description = interpret_and_describe_code(code)
        if debug:
            print("Generated Description:", code_description)

        # Step 2: Generate new code based on the description
        generated_code = generate_code_from_description(code_description, Retrieval)
        generated_code = format_check(generated_code)
        if debug:
            print("Generated Code:\n", generated_code)
        return generated_code
    except Exception as e:
        print("An error occurred:", str(e))
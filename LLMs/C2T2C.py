
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

with open('secret.json') as f:
    api_key = json.load(f)
os.environ["OPENAI_API_KEY"] = api_key["key"]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # insert Change model option!!!
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

prompt_template_c2t = ChatPromptTemplate.from_messages([
    ("system", "You are a expert Python programmer, describe the code to make sure a software engineer can reproduce the code by the description. \n Make sure to mention: \n 1. Function name\n 2. The purpose of the code\n3. The input/output format of the code"),
    ("user", "{input}"),
])
output_parser_c2t = StrOutputParser()
# Define a pipeline combining prompt, model invocation, and output parsing
pipeline_c2t = prompt_template_c2t | llm | output_parser_c2t


prompt_template_t2c = ChatPromptTemplate.from_messages([
    ("system", "You are a expert Python programmer, write a Python code that fits the description I give you. Make sure you only return the code. Do not output ```python and ```"),
    ("user", "{input}"),
])
output_parser_t2c = StrOutputParser()
# Define a pipeline combining prompt, model invocation, and output parsing
pipeline_t2c = prompt_template_t2c | llm | output_parser_t2c

def interpret_and_describe_code(code):
    # Using OpenAI's GPT model to interpret code and generate a description
    input = code
    response = pipeline_c2t.invoke({"input": input})
    return response

def generate_code_from_description(description):
    # Generating code from the description using an OpenAI model
    input = description
    response = pipeline_t2c.invoke({"input": input})
    return response

# Example usage of the functions
example_code = """
import math
def calculate_area(radius):
    return math.pi * radius ** 2
"""

# Step 1: Interpret the code
code_description = interpret_and_describe_code(example_code)
print("Generated Description:", code_description)

# Step 2: Generate new code based on the description
generated_code = generate_code_from_description(code_description)
print("Generated Code:\n", generated_code)

def format_check(result):
    result = result.replace("```python",'')
    result = result.replace("```",'')
    return result

def C2T2C(code, debug = False):
    try:
        code_description = interpret_and_describe_code(code)
        if debug:
            print("Generated Description:", code_description)

        # Step 2: Generate new code based on the description
        generated_code = generate_code_from_description(code_description)
        if debug:
            print("Generated Code:\n", generated_code)
        generated_code = format_check(generated_code)
        return generated_code
    except Exception as e:
        print("An error occurred:", str(e))
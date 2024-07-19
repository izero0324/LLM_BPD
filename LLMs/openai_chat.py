import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLMs.LLM_models import openAI
from tools.prompts import get_system_prompt
from tools.output_cleaner import python_output

# # Read API key from the secret file
# with open('secret.json') as f:
#     data = json.load(f)
# api_key = data["key"]
# langchain_key = data["langchian_api"]

# # Export the API key
# os.environ["OPENAI_API_KEY"] = api_key
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = langchain_key


# class LLM_model:
#     def __init__(self) -> None:
#         # Initialize the LLM with OpenAI
#         _llm = ChatOpenAI(
#             model="gpt-3.5-turbo",  # insert Change model option!!!
#             temperature=0,
#             max_tokens=None,
#             timeout=None,
#             max_retries=2
#         )
#         pass
#     def models(self):
#         # Define a pipeline combining prompt, model invocation, and output parsing
#         return self._llm
# def llm(model_name, temp=0):
#     llm = ChatOpenAI(
#                 model=model_name,  # insert Change model option!!!
#                 temperature=temp,
#                 max_tokens=None,
#                 timeout=None,
#                 max_retries=2
#             )
#     return llm

def optimize_code(code, model_name):
    # Define a prompt to simulate the role of the system
    systme_prompt = get_system_prompt()
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", systme_prompt),            ("user", "{input}"),
        ])
    output_parser = StrOutputParser()
    model = openAI(model_name,temp=0)
    try:
        input = code
        pipeline = prompt_template | model | output_parser
        response = pipeline.invoke({"input": input})
        response = python_output(response)
        return response
    except Exception as e:
        print("An error occurred:", str(e))

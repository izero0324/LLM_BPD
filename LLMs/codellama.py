import json
from transformers import pipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains import SimpleSequentialChain

from tools.output_cleaner import python_output
from LLMs.LLM_models import llamas


concept_prompt_template = """You are a expert Python programmer, and here is your task:\n1. I will give you a python code and you will rewrite to make it execute faster.\n2. Read the function name and continue using it. \n3. Only return the code.\n4. Be sure the return is inside the function\n ---\nThe code to be rewirte is as below: \n{code2op}\n please rewrite the code :"""
    concept_prompt = PromptTemplate(
        input_variables=["code2op"],
        template=concept_prompt_template,
    )

llm = llamas('llama3')
def codellama_optimise(input, model):
    print('model:',model)
    concept_chain = LLMChain(llm=llm, prompt=concept_prompt)
    #input = "def is_Power_Of_Two (x): \r\n    return x and (not(x & (x - 1))) \r\ndef differ_At_One_Bit_Pos(a,b): \r\n    return is_Power_Of_Two(a ^ b)"

    output = concept_chain.invoke(input)
    raw_result = output['text']
    raw_result = raw_result.split('please rewrite the code :')[1]
    # raw_result = raw_result.replace('```python', '```')
    # result = raw_result.replace('```', '').replace('\n\n','')
    result = python_output(raw_result)

    print( result)
    return result


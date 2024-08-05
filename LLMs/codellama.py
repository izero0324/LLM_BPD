import json
from transformers import pipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains import SimpleSequentialChain

from tools.output_cleaner import python_output
from tools.prompts import get_system_prompt
from tools.retrieval import retrive_docs, retrive_codes, retrive_18kcodes
from LLMs.LLM_models import llamas


concept_prompt_template = get_system_prompt('llama')
concept_prompt = PromptTemplate(
    input_variables=["code2op"],
    template=concept_prompt_template,
)


def codellama_optimise(input, model, Retrieval = False):
    print('model:',model)
    llm = llamas(model)
    concept_chain = LLMChain(llm=llm, prompt=concept_prompt)
    #input = "def is_Power_Of_Two (x): \r\n    return x and (not(x & (x - 1))) \r\ndef differ_At_One_Bit_Pos(a,b): \r\n    return is_Power_Of_Two(a ^ b)"
    if Retrieval:
        input += retrive_18kcodes(input, k=1)
        
    output = concept_chain.invoke(input)
    raw_result = output['text']
    raw_result = raw_result.split('please rewrite the code:')[1]
    print("RAW results", raw_result)
    try:
        raw_result = raw_result.split('```python')[1].split('```')[0]
    except:
        raw_result = raw_result
    # raw_result = raw_result.replace('```python', '```')
    # result = raw_result.replace('```', '').replace('\n\n','')
    result = python_output(raw_result)
    print(result)
    return result


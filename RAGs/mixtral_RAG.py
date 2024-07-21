from transformers import pipeline
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain
from langchain.chains import SimpleSequentialChain

from tools.output_cleaner import python_output
from LLMs.LLM_models import mixtral_model
from tools.prompts import get_system_prompt
from tools.retrieval import retrive_docs


concept_prompt_template = get_system_prompt('mixtral')
concept_prompt = PromptTemplate(
    input_variables=["code2op"],
    template=concept_prompt_template,
)

def mixtral_RAG(code, model):
    print('model:', model)
    llm = mixtral_model()
    rd = retrive_docs(code, k=1)
    concept_chain = LLMChain(llm=llm, prompt=concept_prompt)
    #input = "def is_Power_Of_Two (x): \r\n    return x and (not(x & (x - 1))) \r\ndef differ_At_One_Bit_Pos(a,b): \r\n    return is_Power_Of_Two(a ^ b)"
    input = code + rd
    output = concept_chain.invoke(input)
    raw_result = output['text']
    print(raw_result)
    raw_result = raw_result.split('please rewrite the code and only return the code without any explain:')[1]
    #     # raw_result = raw_result.replace('```python', '```')
    #     # result = raw_result.replace('```', '').replace('\n\n','')
    result = python_output(raw_result)

    print( result)
    return result
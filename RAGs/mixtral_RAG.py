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
    input = code + rd
    output = concept_chain.invoke(input)
    raw_result = output['text']
    raw_result = raw_result.split('please rewrite the code and only return the code without any explain:')[1]
    result = python_output(raw_result)
    #print(result)
    return result
from LLMs.openai_chat import optimize_code
from LLMs.naive_rag import use_RAG
from tools.extract import get_data

RAG_dataset = get_data('test')

def LLM_revise(code, model):

    if model=='GPT3.5':
        result = optimize_code(code)
    elif model=='naive RAG':
        result = use_RAG(RAG_dataset, code)
    else:
        raise BaseException(f"{model} not a valid model. Choose from 'GPT3.5' and 'naive RAG'" )
    return result
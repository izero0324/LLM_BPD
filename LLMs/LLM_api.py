from LLMs.openai_chat import optimize_code
#from LLMs.RAG_python import optimize_code_FAISS
from LLMs.C2T2C import C2T2C
from tools.extract import get_data

RAG_dataset = get_data('train')

def LLM_revise(code, model, debug):

    if model=='GPT3.5':
        result = optimize_code(code)
    elif model=='naive RAG':
        from LLMs.naive_rag import use_RAG
        result = use_RAG(RAG_dataset, code)
    # elif model == 'OpenAI EMB':
    #     result = optimize_code_FAISS(code)
    elif model == 'C2T2C':
        result =  C2T2C(code, debug)
    else:
        raise BaseException(f"{model} not a valid model. Choose from 'GPT3.5' and 'naive RAG'" )
    return result
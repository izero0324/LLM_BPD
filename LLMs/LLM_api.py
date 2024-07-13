from LLMs.openai_chat import optimize_code
#from LLMs.RAG_python import optimize_code_FAISS
from RAGs.C2T2C import C2T2C
from LLMs.codellama import codellama_optimise
from LLMs.mixtral import mixtral_gen
from tools.extract import get_data


RAG_dataset = get_data('train')

def LLM_revise(code, model, debug):

    if model=='gpt-3.5-turbo':
        result = optimize_code(code,model)
    elif model == 'gpt-4-turbo':
        result = optimize_code(code, model)
    elif model=='naive RAG':
        from LLMs.naive_rag import use_RAG
        result = use_RAG(RAG_dataset, code)
    # elif model == 'OpenAI EMB':
    #     result = optimize_code_FAISS(code)
    elif model == 'C2T2C':
        result =  C2T2C(code, debug)
    elif model == 'codellama' or 'codellama-13b':
        result = codellama_optimise(code, model)
    elif model == 'mixtral':
        result = mixtral_gen(code, model)
    else:
        raise BaseException(f"{model} not a valid model. Choose from 'GPT3.5' and 'naive RAG'" )
    return result
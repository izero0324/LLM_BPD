from LLMs.openai_chat import optimize_code
#from LLMs.RAG_python import optimize_code_FAISS
from RAGs.C2T2C import C2T2C
from RAGs.iterative_RAG import iterative_RAG_gen
from LLMs.codellama import codellama_optimise
from LLMs.mixtral import mixtral_gen
from tools.extract import get_data


RAG_dataset = get_data('train')

def LLM_revise(code, model, debug):

    if 'gpt' in model:
        result = optimize_code(code,model)
    elif model=='naive RAG':
        from LLMs.naive_rag import use_RAG
        result = use_RAG(RAG_dataset, code)
    elif model == 'mixtral_RAG':
        from RAGs.mixtral_RAG import mixtral_RAG
        result = mixtral_RAG(code, model)
    # elif model == 'OpenAI EMB':
    #     result = optimize_code_FAISS(code)
    elif model == 'C2T2C':
        result =  C2T2C(code, debug)
    elif model == 'codellama' or 'codellama-13b' or 'llama3':
        result = codellama_optimise(code, model)
    elif model == 'codellama_RAG':
        result = codellama_optimise(code,'llama3' , Retrieval=True)
    elif model == 'llama3_RAG':
        result = codellama_optimise(code,'llama3' , Retrieval=True)
    elif model == 'mixtral':
        result = mixtral_gen(code, model)
    elif model == 'iter':
        result = iterative_RAG_gen(code,model)
    else:
        raise BaseException(f"{model} not a valid model. Choose from 'GPT3.5' and 'naive RAG'" )
    return result
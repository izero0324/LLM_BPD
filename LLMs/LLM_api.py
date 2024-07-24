from LLMs.openai_chat import optimize_code
#from LLMs.RAG_python import optimize_code_FAISS
from RAGs.CCG_RAG import CCG_RAG
from RAGs.iterative_RAG import iterative_RAG_gen
from LLMs.codellama import codellama_optimise
from LLMs.mixtral import mixtral_gen
from tools.extract import get_data

#self search depricated
#RAG_dataset = get_data('train')

def LLM_revise(code, model, debug):

    # GPT series
    if 'gpt' in model:
        if 'RAG' in model:
            model = model.replace('_RAG', '')
            result = optimize_code(code, model, Retrieval=True)
        else:
            result = optimize_code(code,model)
    
    # Mixtral series
    elif 'mixtral' in model:
        if 'RAG' in model:
            model = model.replace('_RAG', '')
            from RAGs.mixtral_RAG import mixtral_RAG
            result = mixtral_RAG(code, model)
        else:
            result = mixtral_gen(code, model)

    # llama series
    elif 'llama' in model:
        if 'RAG' in model:
            model = model.replace('_RAG', '')
            result = codellama_optimise(code, model , Retrieval=True)
        else:
            result = codellama_optimise(code, model)
            print('By codellamas')
    
    # CCG-RAG : CCG_RAG
    elif 'CCG_RAG' in model:
        if 'instruct' in model:
            result = CCG_RAG(code, debug, Retrieval= True)
        else:
            result =  CCG_RAG(code, debug)
    
    # iterative RAG
    elif model == 'iter':
        print("(((((((((((((((((((((((((((Iterative!!)))))))))))))))))))))))))))")
        result = iterative_RAG_gen(code,model)
    else:
        print(f'==============={model} invalid =================')
        raise BaseException(f"{model} not a valid model. Choose from 'GPT3.5' and 'naive RAG'" )
    return result
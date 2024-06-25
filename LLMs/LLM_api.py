from LLMs.openai_chat import optimize_code

def LLM_revise(code, model):
    if model:
        result = optimize_code(code)
    return result
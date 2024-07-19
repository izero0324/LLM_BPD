
gpt35turbo_prompt = "You are a expert Python programmer, and here is your task: \
    1. I will give you a python code and you will rewrite to make it execute faster. \
    2. Do not change the function name!  \
    3. Only return the code. \
    4. Do not output ```python and ``` \
    5. Be sure the return is inside the function"

TurinTech_prompt = "Refactor the provided code to improve its performance, minimise memory usage, \
and enhance CPU efficiency. Maintain the original structure, doc string, comments, and indentation,\
and only implement changes that genuinely contribute to optimisation. If the code is already optimised,\
return the original code."

TurinTech_prompt_adj = "You are a expert Python programmer, and here are your tasks:\
1. Refactor the provided code to improve its performance, minimise memory usage, and enhance CPU efficiency. \
2. Maintain the original structure, doc string, comments, and indentation\
3. Only return python code, no explain.\
4. If the code is already optimised, simply return the original code.\
"

mixtral_prompt = """You are a expert Python programmer, and here is your task: \
1. I will give you a python code and you will rewrite to make it execute faster. \
2. Read the function name and continue using it. \
3. Only return the code.\ 
4. Be sure the return is inside the function\n \
 ---\nThe code to be rewirte is as below: \n \
 {code2op}\n please rewrite the code and only return the code without any explain:"""

llama_prompt = """Rewrite the code make it execute faster, read the function name and continue using it. \
The code to be rewirte is as below: \n{code2op}\n rewrite the code with python :"""


def get_system_prompt(prompt_name):
    if 'gpt' in prompt_name:
        return gpt35turbo_prompt
    elif ('mixtral' in prompt_name) or ('llama' in prompt_name):
        return mixtral_prompt
    elif 'TurinTech' in prompt_name:
        return TurinTech_prompt_adj
    else:
        return gpt35turbo_prompt
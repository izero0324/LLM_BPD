import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from LLMs.mixtral import mixtral_gen
import LLMs.LLM_models as BaseModels
from LLMs.openai_chat import optimize_code
from RAGs.CCG_RAG import CCG_RAG
from tools.pylint_check import check_pylint


# Initialize the LLM with OpenAI GPT model
error_fix_llm = BaseModels.openAI('gpt-3.5-turbo')
prompt_template_error_fix= ChatPromptTemplate.from_messages([
    ("system", "You are a expert Python programmer, Fix these errors in the following python codes."),
    ("user", "code: {ori_code} \n Errors: {code_error}."),
    ("user", "Please Start fixing the code, only return the fixed code:")
])
output_parser_error_fix = StrOutputParser()
pipeline_error_fix = prompt_template_error_fix | error_fix_llm | output_parser_error_fix

def regenerate_code(message, code):
    # Generating code from the description using an OpenAI model
    response = pipeline_error_fix.invoke({"ori_code": code, "code_error": message})
    try:
        code_response = response.split("only return the fixed code:")[1]
    except:
        code_response = response
    try:
        code_response = code_response.split("System:")[1].split("Human:")[0]
    except:
        code_response = code_response
    try:
        final_response = code_response.split("```python")[1].split("```")[0]
    except:
        final_response = code_response
    #final_response = code_response.split("```python")[1].split("```")[0]
    return final_response


# Sample code for interpretation and evaluation
def interpret_code(code):
    """ Interpret provided code, return True if runs without errors else False """
    try:
        exec(code)
        return True, "Code executed successfully"
    except Exception as e:
        return False, str(e)


def iterative_RAG_gen(initial_code,model_name):
    print(model_name, " :")
    #first_result = mixtral_gen(initial_code, 'mixtral for iterGAN')
    first_result = CCG_RAG(initial_code)
    first_result = optimize_code(initial_code, 'gpt-3.5-turbo')
    correct, message = interpret_code(first_result)
    print('============Iter_check_point============\n',first_result, 'Correct? \n', correct)
    
    if correct:
        suggestion = first_result

    if not correct:
        print(f"Error: {message}")
        suggestion = regenerate_code(message, first_result)
        correct, message = interpret_code(suggestion)
    
    score, messages = check_pylint(suggestion)
    if score == 0:
        return suggestion
    else:
        message = ''.join(messages)
        suggestion = regenerate_code(message, first_result)

    print("Fixed code: ", suggestion)
    return suggestion
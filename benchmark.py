from tools.extract import get_data, extract_code
from tools.python_exec import test_code
from LLMs.LLM_api import LLM_revise


dataset = get_data("few-shot")
accuracy = 0
boost = 0
counter = 0

for test_data in dataset:
    code = extract_code(test_data)
    # original run
    success, runtime, error = test_code(test_data)
    
    # pass code to LLM for optimize
    test_data['code'] = LLM_revise(code, model= 'GPT3.5')
    LLM_success, LLM_runtime, LLM_error = test_code(test_data)

    accurate = (LLM_success==1)
    effi_boost = (runtime - LLM_runtime)/LLM_runtime > 0  # Modify to % with threashold
    
    accuracy += accurate
    boost += effi_boost
    # print("Original cade: ", success, runtime, error)
    # print("ChatGPT: ",LLM_success, LLM_runtime, LLM_error)
    counter += 1
    if counter%10 == 0:
        print(counter/10, "0%", " done")


accuracy /= len(dataset) 

print("accuracy: ",accuracy * 100,"Code boosted: ", boost)

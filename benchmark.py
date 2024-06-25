from extract import get_data, extract_code
from python_exec import test_code
from LLM_api import LLM_revise


dataset = get_data("few-shot")
accuracy = 0
boost = 0

for test_data in dataset:
    code = extract_code(test_data)
    # original run
    success, runtime, error = test_code(test_data)
    
    # pass code to LLM for optimize
    test_data['code'] = LLM_revise(code)
    LLM_success, LLM_runtime, LLM_error = test_code(test_data)

    accurate = (LLM_success==1)
    effi_boost = (runtime - LLM_runtime)/LLM_runtime > 0.15  # Modify to % with threashold
    
    accuracy += accurate
    boost += effi_boost

accuracy /= len(dataset) 

print(accuracy * 100, boost)

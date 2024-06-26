import argparse
from tools.extract import get_data, extract_code
from tools.python_exec import test_code
from tools.bleu import code_bleu
from LLMs.LLM_api import LLM_revise

def benchmark_process(dataset, model, debug = False):
    accuracy = 0 
    boost= 0 
    mem_reduce = 0
    flake8 = 0
    bleu_sum = 0

    for test_data in dataset:

        code = extract_code(test_data)
        # original run
        success, runtime, error, flake8_error, mem_kb = test_code(test_data)
        
        # pass code to LLM for optimize
        test_data['code'] = LLM_revise(code, model = model)
        LLM_success, LLM_runtime, LLM_error, LLM_flake8_error, LLM_mem_kb = test_code(test_data)

        accurate = (LLM_success==1)
        effi_boost = (runtime - LLM_runtime)/LLM_runtime > 0  # Modify to % with threashold
        
        BLEU = code_bleu(code, test_data['code'])
        bleu_sum += BLEU
        accuracy += accurate
        boost += effi_boost
        mem_reduce += (mem_kb - LLM_mem_kb)>0
        flake8 += int(flake8_error) - int(LLM_flake8_error)
        if debug:
            print("Original cade: ", success, runtime, error, flake8_error, mem_kb)
            print("ChatGPT: ",LLM_success, LLM_runtime, LLM_error, LLM_flake8_error, LLM_mem_kb)
            print("BLEU: ", BLEU)
        # counter += 1
        # if counter%10 == 0:
        #     print(counter/10, "0%", " done")
    accuracy /= len(dataset) 
    bleu_sum /= len(dataset)
    print("accuracy: ",accuracy * 100,"Code boosted: ", boost,"Memory reduced: ", mem_reduce,  "flake8 fixed: ", flake8, "BLEU: ", bleu_sum)
    return accuracy * 100, boost, mem_reduce, flake8, bleu_sum


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run an experiment with specified parameters.')
    parser.add_argument('--data', type=str, default="few-shot", help='dataset name')
    parser.add_argument('--model', type=str, default='GPT3.5', help='model name.')
    parser.add_argument('--debug', type=bool, default=False, help='Debug mode')

    return parser.parse_args()


def main():
    '''
    Entry point
    '''
    args = parse_arguments()
    benchmark_process(dataset= get_data(args.data), model=args.model, debug=args.debug)
    

if __name__ == '__main__':
    print("==================================================")
    print("Start Python code optimize evaluation benchmark...")
    print("==================================================")
    main()

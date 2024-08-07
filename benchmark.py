import argparse
from tools.extract import get_data, extract_code
from tools.python_exec import test_code
from tools.bleu import code_bleu
from tools.codeMatrix import codeMatrix_improve
from LLMs.LLM_api import LLM_revise


def benchmark_process(dataset, model, debug = False):
    accuracy = 0
    boost= 0
    mem_reduce = 0
    flake8 = 0
    bleu_sum = 0
    total_flake8 = 0
    counter = 0
    Cyclomatic = 0
    Halstead = 0
    already_optimised = 0

    for test_data in dataset:

        code = extract_code(test_data)
        # original run
        success, runtime, error, flake8_error, mem_kb = test_code(test_data)

        # pass code to LLM for optimize
        optimised_code = LLM_revise(code, model = model, debug=debug)
        if test_data['code'] == optimised_code: #check if the code didn't change
            already_optimised += 1
        
        test_data['code'] = optimised_code
        LLM_success, LLM_runtime, LLM_error, LLM_flake8_error, LLM_mem_kb = test_code(test_data)

        accurate = (LLM_success==1)
        effi_boost = (runtime - LLM_runtime)/LLM_runtime > 0  # Modify to % with threashold

        if LLM_success ==1:
            BLEU = code_bleu(code, test_data['code'])
            H_improve , cc_imporve = codeMatrix_improve(code, test_data['code'])
            Cyclomatic += cc_imporve
            Halstead += H_improve
            bleu_sum += BLEU
            accuracy += accurate
            boost += effi_boost
            mem_reduce += (mem_kb - LLM_mem_kb)>0
            flake8 += int(flake8_error) - int(LLM_flake8_error)
        else:
            BLEU = 0
            cc_imporve = False
            H_improve = False
        total_flake8 += int(flake8_error)
        if debug:
            print("Original code: ", success, runtime, error, flake8_error, mem_kb)
            print(model, ": ",LLM_success, LLM_runtime, LLM_error, LLM_flake8_error, LLM_mem_kb)
            print("BLEU: ", BLEU, "Cyclomatic: ", cc_imporve, "Halstead: ", H_improve)
        counter += 1
        if counter%10 == 0:
            print('==============Check Point ==============')
            print(counter," / ", len(dataset), " done")
            ck_acc = accuracy / counter 
            ck_bleu = bleu_sum / counter
            print("accuracy: ",ck_acc * 100, "Code boosted: ", boost, "/", counter, "Memory reduced: ",
            mem_reduce,  "flake8 fixed: ", flake8, "/", total_flake8, "BLEU: ", ck_bleu, 
            "Cyclomatic: ", Cyclomatic, "Halstead: ", Halstead,
            "Can't optimise: ", already_optimised)
    accuracy /= len(dataset) 
    bleu_sum /= len(dataset)
    print("accuracy: ",accuracy * 100, "Code boosted: ", boost, "/", len(dataset), "Memory reduced: ",
     mem_reduce,  "flake8 fixed: ", flake8, "/", total_flake8, "BLEU: ", bleu_sum,
     "Cyclomatic: ", Cyclomatic, "Halstead: ", Halstead,
    "Can't optimise: ", already_optimised)
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

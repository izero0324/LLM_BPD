import json

def get_data(subset):
    with open('./dataset/mbpp.jsonl') as f:
        data = [json.loads(line) for line in f]

    # Choose partition
    if subset == "few-shot":
        eval_data = data[:10]
    elif subset ==  "test":
        eval_data = data[10:510]
    elif subset == "validation":
        eval_data = data[510:600]
    elif subset == "train":
        eval_data = data[600:]
    elif subset == "continue":
        eval_data = data[960:]
    return eval_data

def extract_code(data):
    return data['code']

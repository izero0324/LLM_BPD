from transformers import LlamaForCausalLM, CodeLlamaTokenizer

tokenizer = CodeLlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
model = LlamaForCausalLM.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")
PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"]
generated_ids = model.generate(input_ids, max_new_tokens=128)

filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
print(PROMPT.replace("<FILL_ME>", filling))
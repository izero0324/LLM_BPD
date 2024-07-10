from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers

checkpoint = "Salesforce/codegen-350M-mono"
# # Initializing a CodeGen 6B configuration
# configuration = CodeGenConfig()
# # Initializing a model (with random weights) from the configuration
# model = CodeGenModel(configuration)
# # Accessing the model configuration
# configuration = model.config

checkpoint = "Salesforce/codegen-350M-mono"
config = AutoConfig.from_pretrained(
  checkpoint,
  max_new_tokens=1024
)

model = AutoModelForCausalLM.from_config(config)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "def hello_world():"


input = "def is_Power_Of_Two (x): \r\n    return x and (not(x & (x - 1))) \r\ndef differ_At_One_Bit_Pos(a,b): \r\n    return is_Power_Of_Two(a ^ b)"


# Example query and documents
query = "You are a expert Python programmer, and here is your task:\n 1. I will give you a python code and you will rewrite to make it execute faster.\n 2. Do not change the function name!  \n 3. Only return the code.\n 4. Do not output ```python and ```\n 5. Be sure the return is inside the function\n --- \nHere's the code to be optimized:"
generate_code = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    top_p=0.15,  # select from top tokens whose probability add up to 15%
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    max_new_tokens=500,  # mex number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)
#completion = model.generate(**tokenizer(query+input, return_tensors="pt"))
completion = generate_code(query+input)
print(tokenizer.decode(completion[0]['generated_text']))
#print(completion)
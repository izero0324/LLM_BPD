from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import CodeGenConfig, CodeGenModel


checkpoint = "Salesforce/codegen-350M-mono"
# # Initializing a CodeGen 6B configuration
# configuration = CodeGenConfig()
# # Initializing a model (with random weights) from the configuration
# model = CodeGenModel(configuration)
# # Accessing the model configuration
# configuration = model.config

checkpoint = "Salesforce/codegen-350M-mono"
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

text = "def hello_world():"


input = "def is_Power_Of_Two (x): \r\n    return x and (not(x & (x - 1))) \r\ndef differ_At_One_Bit_Pos(a,b): \r\n    return is_Power_Of_Two(a ^ b)"


# Example query and documents
query = "You are a expert Python programmer, and here is your task:\n 1. I will give you a python code and you will rewrite to make it execute faster.\n 2. Do not change the function name!  \n 3. Only return the code.\n 4. Do not output ```python and ```\n 5. Be sure the return is inside the function\n --- \nHere's the code to be optimized:"

completion = model.generate(**tokenizer(query+input, return_tensors="pt"))

print(tokenizer.decode(completion[0]))
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from typing import List

# Load the model and tokenizer locally
model_name = "codellama/CodeLlama-13b-hf" # Replace with your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Adjusting the device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom LLM class to interface with Langchain
class LocalHuggingFaceLLM(LLM):
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs['input_ids'], 
                                      max_new_tokens=4096, 
                                      temperature=0)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

# Instantiate the custom LLM
local_llm = LocalHuggingFaceLLM(model=model, tokenizer=tokenizer, device=device)

# Define your prompt template
concept_prompt = PromptTemplate(template="Your input template with placeholders if any.")

# Create LLMChain with the local LLM
llm_chain = LLMChain(llm=local_llm, prompt=concept_prompt)

# Use the LLMChain to generate a response
input_prompt = "Your input text here"
filled_template = concept_prompt.fill_text(**{"input": input_prompt})
response = llm_chain.predict(filled_template)
print(response)
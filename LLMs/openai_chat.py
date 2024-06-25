import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Read API key from the secret file
with open('secret.json') as f:
    data = json.load(f)
api_key = data["key"]
langchain_key = data["langchian_api"]

# Export the API key
os.environ["OPENAI_API_KEY"] = api_key
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = langchain_key

# Initialize the LLM with OpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Define a prompt to simulate the role of the system
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a professional python developer, I will give you a code and you will make it run faster.\n 1. Do not change the function name!\n 2. Only return the code.\n 3. Do not output ```python and ```"),
    ("user", "{input}"),
])

output_parser = StrOutputParser()

# Define a pipeline combining prompt, model invocation, and output parsing
pipeline = prompt_template | llm | output_parser

def optimize_code(code):
    try:
        input = code
        response = pipeline.invoke({"input": input})
        return response
    except Exception as e:
        print("An error occurred:", str(e))

def chat_with_bot(initial_message="Hello, I'm your assistant. How can I help you today?"):
    print(initial_message)
    while True:
        try:
            user_input = input("You: ")  # Get user input
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Exiting chat...")
                break

            response = pipeline.invoke({"input": user_input})
            print("Bot:", response)
        except Exception as e:
            print("An error occurred:", str(e))
            break

# Execute the chat session
if __name__ == "__main__":
    chat_with_bot()
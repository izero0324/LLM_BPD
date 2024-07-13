pip3 install -r requirements.txt
touch secret.json
echo "{"key":"openAI_key","langchian_api" : "langchain_key","huggingface_api" : "huggingface key"}" > secret.json
huggingface-cli download meta-llama/Meta-Llama-3-8B --include "original/*" --local-dir Meta-Llama-3-8B --token "Your HuggingFace token here"
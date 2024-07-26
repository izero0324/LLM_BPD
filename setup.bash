pip3 install -r requirements.txt
touch secret.json
echo "{"key":"openAI_key","langchian_api" : "langchain_key","huggingface_api" : "huggingface key"}" > secret.json
export TOKENIZERS_PARALLELISM= false
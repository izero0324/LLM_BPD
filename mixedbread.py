from sentence_transformers import CrossEncoder
from tools.extract import get_data

# Load the model, here we use our base sized model
model = CrossEncoder("mixedbread-ai/mxbai-rerank-base-v1")
input = "def is_Power_Of_Two (x): \r\n    return x and (not(x & (x - 1))) \r\ndef differ_At_One_Bit_Pos(a,b): \r\n    return is_Power_Of_Two(a ^ b)"


# Example query and documents
query = "You are a expert Python programmer, and here is your task:\n 1. I will give you a python code and you will rewrite to make it execute faster.\n 2. Do not change the function name!  \n 3. Only return the code.\n 4. Do not output ```python and ```\n 5. Be sure the return is inside the function\n --- \nHere's the code to be optimized:"
# documents = [
#     "'To Kill a Mockingbird' is a novel by Harper Lee published in 1960. It was immediately successful, winning the Pulitzer Prize, and has become a classic of modern American literature.",
#     "The novel 'Moby-Dick' was written by Herman Melville and first published in 1851. It is considered a masterpiece of American literature and deals with complex themes of obsession, revenge, and the conflict between good and evil.",
#     "Harper Lee, an American novelist widely known for her novel 'To Kill a Mockingbird', was born in 1926 in Monroeville, Alabama. She received the Pulitzer Prize for Fiction in 1961.",
#     "Jane Austen was an English novelist known primarily for her six major novels, which interpret, critique and comment upon the British landed gentry at the end of the 18th century.",
#     "The 'Harry Potter' series, which consists of seven fantasy novels written by British author J.K. Rowling, is among the most popular and critically acclaimed books of the modern era.",
#     "'The Great Gatsby', a novel written by American author F. Scott Fitzgerald, was published in 1925. The story is set in the Jazz Age and follows the life of millionaire Jay Gatsby and his pursuit of Daisy Buchanan."
# ]
documents = str(get_data('train'))

query+=input
# Lets get the scores
results = model.rank(query, documents, return_documents=True, top_k=3)
#results = model.generate(query)
print(results)
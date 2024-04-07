

from langchain.text_splitter import CharacterTextSplitter

with open("./data/data_transformer.txt", encoding='UTF-8') as f:
    open_ai_description = f.read()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=40, chunk_overlap=10, separator='。'
)
texts = text_splitter.split_text(open_ai_description)

for t in texts:
    print(t)


# print(texts)

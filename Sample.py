from pathlib import Path
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index import download_loader,LLMPredictor
from langchain.chat_models import ChatOpenAI

# Windows 11 Pro 22H2
# Python 3.11.1
# pip 22.3.1
# openai 0.27.6
# langchain 0.0.158

# 前提!
# 環境変数OPENAI_API_KEYにAPIKeyをセットしておく

# インプットとなるcsvを用意する
csvfile = Path('D:\Work\Projects\ChatGPT\sample.csv')

SimpleCSVReader = download_loader("SimpleCSVReader")
loader = SimpleCSVReader()
documents = loader.load_data(file=csvfile)

llm_predictor = LLMPredictor(
    llm=ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )
)

index = GPTVectorStoreIndex.from_documents(
    documents,
    llm_predictor=llm_predictor
)

query_engine = index.as_query_engine()

# 質問を入力する
question = input("質問を入力してください: ")
answer = query_engine.query(question)
print(answer)

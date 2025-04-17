from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.cosmos import CosmosClient
from spellchecker import SpellChecker
from cachetools import cached, TTLCache
from langchain.llms import GPT4All
from langchain.chat_models import OpenAIClient
from bs4 import BeautifulSoup
import requests
import os
import json

app = FastAPI()
# Define a model for the user input
# ユーザー入力のモデルを定義
class UserInput(BaseModel):
    question: str

# Initialize CosmosDB client
# CosmosDBクライアントを初期化
cosmos_client = CosmosClient(os.getenv('COSMOS_ENDPOINT'), os.getenv('COSMOS_KEY'))
database_name = 'your-database-name'
container_name = 'your-container-name'
database = cosmos_client.get_database_client(database_name)
container = database.get_container_client(container_name)

spell = SpellChecker()
dataset = load_dataset('medical_dialog', 'processed.en')
df = pd.DataFrame(dataset['train'])

dialog = []

# 患者と医者の発言をそれぞれ抽出した後、順にリストに格納
# Extract patient and doctor statements and store them in a list sequentially
patient, doctor = zip(*df['utterances'])
for i in range(len(patient)):
  dialog.append(patient[i])
  dialog.append(doctor[i])

df_dialog = pd.DataFrame({"dialog": dialog})

# 成形終了したデータセットを保存
# Save the formatted dataset
df_dialog.to_csv('medical_data.txt', sep=' ', index=False)

loader = TextLoader('medical_data.txt', encoding="utf-8")
index = VectorstoreIndexCreator(embedding= HuggingFaceEmbeddings()).from_loaders([loader])

# Add a configuration setting for LLM selection
# LLMの選択に基づいて初期化
# デフォルトは 'gpt4all'
llm_choice = os.getenv('LLM_CHOICE', 'gpt4all')

# Modify LLM initialization based on the choice
if llm_choice == 'gpt4all':
    # GPT4Allモデルの初期化
    llm_path = './model/ggml-gpt4all-j-v1.3-groovy.bin'
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=llm_path, callbacks=callbacks, verbose=True, backend='gptj')
elif llm_choice == 'openai':
    # Example for OpenAI LLM initialization
    llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
else:
    raise ValueError(f"Unsupported LLM choice: {llm_choice}")

template = """
Please use the following context to answer questions.
Context: {context}
---
Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["context", "question"]).partial(context=context)

# Update the LLMChain initialization
llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.run("what is the solution for soar throat"))

def preprocess_input(question):
    # Correct spelling
    corrected_question = " ".join([spell.correction(word) for word in question.split()])
    return corrected_question

# Function to call Azure OpenAI API
# Azure OpenAI APIを呼び出す関数
async def call_openai_api(prompt):
    try:
        # Define the API endpoint and headers
        api_url = "https://<your-openai-endpoint>/openai/deployments/<deployment-id>/completions?api-version=2022-12-01"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }

        # Define the payload
        payload = {
            "prompt": prompt,
            "max_tokens": 150
        }

        # Make the POST request
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        # Parse the response
        result = response.json()
        return result['choices'][0]['text']

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API call failed: {str(e)}")

def validate_input(question):
    if not question or len(question) > 256:
        raise HTTPException(status_code=400, detail="Invalid input")

# キャッシュを設定（TTL: 10分）
# Set cache (TTL: 10 minutes)
cache = TTLCache(maxsize=100, ttl=600)

@cached(cache)
def search_data_with_cache(question):
    return search_data_in_cosmosdb(question)

# Define a set of test questions and expected answers
# テスト質問と期待される回答のセットを定義
TEST_QUESTIONS = [
    {"question": "What is the solution for sore throat?", "expected_answer": "Gargle with warm salt water."},
    # Add more test questions and expected answers here
]

# Function to evaluate response accuracy
# 応答の精度を評価する関数
def evaluate_response_accuracy(response, expected_answer):
    # Simple string comparison for demonstration purposes
    # デモ用の単純な文字列比較
    return response.strip().lower() == expected_answer.strip().lower()

# Endpoint to handle user input
# ユーザー入力を処理するエンドポイント
@app.post("/ask")
async def ask_question(user_input: UserInput):
    try:
        # 入力のバリデーション
        # 質問が空でないか、長すぎないかを確認
        validate_input(user_input.question)
        
        # データの検索
        # 質問に関連するデータをキャッシュを利用して検索
        items = search_data_with_cache(user_input.question)
        
        if not items:
            return {"answer": "I'm sorry, I couldn't find any relevant information. Can I help you with something else?"}
        
        # 取得したデータを使用して応答を生成
        # 検索結果を基にAIが応答を生成
        response_data = items[0]
        prompt = f"User question: {user_input.question}\nRelevant data: {response_data}\nGenerate a response."
        ai_response = await call_openai_api(prompt)
        
        # 応答の精度を評価
        # テスト質問に対する応答が期待通りかを確認
        for test in TEST_QUESTIONS:
            if user_input.question == test["question"]:
                if not evaluate_response_accuracy(ai_response, test["expected_answer"]):
                    # 精度が低い場合はLLMを切り替える
                    if llm_choice == 'gpt4all':
                        llm_choice = 'openai'
                    else:
                        llm_choice = 'gpt4all'
                    # LLMを再初期化
                    if llm_choice == 'gpt4all':
                        llm_path = './model/ggml-gpt4all-j-v1.3-groovy.bin'
                        callbacks = [StreamingStdOutCallbackHandler()]
                        llm = GPT4All(model=llm_path, callbacks=callbacks, verbose=True, backend='gptj')
                    elif llm_choice == 'openai':
                        llm = OpenAIClient(api_key=os.getenv('OPENAI_API_KEY'))
                    break
        
        return {"answer": ai_response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
# ヘルスチェックエンドポイント
@app.get("/")
async def root():
    return {"message": "AI Chat System is running"}

def fact_check(statement):
    # ウェブ検索を行う
    search_url = f"https://www.google.com/search?q={statement}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    # 検索結果を解析
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('div', class_='BNeawe vvjwJb AP7Wnd')
    
    # 結果を評価
    for result in results:
        print(result.get_text())

# 例として、ファクトチェックを行う
fact_check("Pythonは最も人気のあるプログラミング言語です")

# Endpoint for Customer Support Assistance
@app.post("/customer_support")
async def customer_support(user_input: UserInput):
    # Implement logic for customer support assistance
    return {"message": "Customer support assistance is not yet implemented."}

# Endpoint for Internal Information Inquiry Support
@app.post("/internal_info_inquiry")
async def internal_info_inquiry(user_input: UserInput):
    # Implement logic for internal information inquiry
    return {"message": "Internal information inquiry support is not yet implemented."}

# Endpoint for Patent Research Support
@app.post("/patent_research")
async def patent_research(user_input: UserInput):
    # Implement logic for patent research support
    return {"message": "Patent research support is not yet implemented."}

# Endpoint for Product Manual Creation Support
@app.post("/product_manual_creation")
async def product_manual_creation(user_input: UserInput):
    # Implement logic for product manual creation
    return {"message": "Product manual creation support is not yet implemented."}

# Endpoint for Legal Support
@app.post("/legal_support")
async def legal_support(user_input: UserInput):
    # Implement logic for legal support
    return {"message": "Legal support is not yet implemented."}

# Endpoint for Travel Planning Support
@app.post("/travel_planning")
async def travel_planning(user_input: UserInput):
    # Implement logic for travel planning support
    return {"message": "Travel planning support is not yet implemented."}

# Endpoint for Medical Diagnosis Support
@app.post("/medical_diagnosis")
async def medical_diagnosis(user_input: UserInput):
    # Implement logic for medical diagnosis support
    return {"message": "Medical diagnosis support is not yet implemented."}

# Endpoint for Academic Research Support
@app.post("/academic_research")
async def academic_research(user_input: UserInput):
    # Implement logic for academic research support
    return {"message": "Academic research support is not yet implemented."}

# Endpoint for Coding Support
@app.post("/coding_support")
async def coding_support(user_input: UserInput):
    # Implement logic for coding support
    return {"message": "Coding support is not yet implemented."} 
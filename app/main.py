from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from azure.cosmos import CosmosClient
from spellchecker import SpellChecker
from cachetools import cached, TTLCache
from langchain.llms import GPT4All
from langchain.chat_models import OpenAIClient
from bs4 import BeautifulSoup
from datasets import load_dataset
import requests
import os
import json
from abc import ABC, abstractmethod
import azure.storage.blob as azure_blob
from azure.identity import DefaultAzureCredential
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# FastAPIアプリケーションのインスタンスを作成
# FastAPIを使用してWebアプリケーションを構築するためのインスタンス
app = FastAPI()

# ユーザー入力のモデルを定義
# ユーザーからの質問を受け取るためのデータモデル
class UserInput(BaseModel):
    question: str

# Initialize CosmosDB client
# CosmosDBクライアントを初期化
# Azure CosmosDBに接続するためのクライアントを設定
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
        # APIエンドポイントとヘッダーを定義
        api_url = "https://<your-openai-endpoint>/openai/deployments/<deployment-id>/completions?api-version=2022-12-01"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }

        # Define the payload
        # リクエストのペイロードを定義
        payload = {
            "prompt": prompt,
            "max_tokens": 150
        }

        # Make the POST request
        # POSTリクエストを送信
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        # Parse the response
        # レスポンスを解析
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
# キャッシュを利用してデータを検索する関数
# 質問に関連するデータをCosmosDBから検索
# キャッシュを利用して効率的にデータを取得
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
    try:
        # Validate the input
        validate_input(user_input.question)

        # Preprocess the input
        processed_question = preprocess_input(user_input.question)

        # Search for relevant internal information using cache
        items = search_data_with_cache(processed_question)

        if not items:
            return {"answer": "I'm sorry, I couldn't find any relevant internal information."}

        # Use the first item from the search results to generate a response
        response_data = items[0]
        prompt = f"User question: {processed_question}\nRelevant data: {response_data}\nGenerate a response."
        ai_response = await call_openai_api(prompt)

        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Patent Research Support
@app.post("/patent_research")
async def patent_research(user_input: UserInput):
    try:
        # Validate the input
        validate_input(user_input.question)

        # Preprocess the input
        processed_question = preprocess_input(user_input.question)

        # Search for relevant patent data using cache
        items = search_data_with_cache(processed_question)

        if not items:
            return {"answer": "I'm sorry, I couldn't find any relevant patent information."}

        # Use the first item from the search results to generate a response
        response_data = items[0]
        prompt = f"User question: {processed_question}\nRelevant data: {response_data}\nGenerate a response."
        ai_response = await call_openai_api(prompt)

        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_product_manual(data):
    # Placeholder logic for generating a product manual
    manual_content = f"Product Manual for {data['product_name']}\n\nFeatures:\n{data['features']}\n\nInstructions:\n{data['instructions']}"
    return manual_content

async def product_manual_creation(user_input: UserInput):
    try:
        # Validate the input
        validate_input(user_input.question)

        # Example input processing (assuming JSON input)
        input_data = json.loads(user_input.question)

        # Generate the product manual
        manual_content = generate_product_manual(input_data)

        # Save the manual to a file (or database)
        manual_filename = f"manuals/{input_data['product_name']}_manual.txt"
        with open(manual_filename, 'w') as manual_file:
            manual_file.write(manual_content)

        # Return a success response
        return {"message": "Product manual created successfully.", "manual_location": manual_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LegalSupportService(SupportService):
    async def handle_request(self, user_input: UserInput):
        # Legal support logic
        return {"message": "Legal support is not yet implemented."}

class TravelPlanningSupportService(SupportService):
    async def handle_request(self, user_input: UserInput):
        try:
            # Validate the input
            validate_input(user_input.question)

            # Preprocess the input
            processed_question = preprocess_input(user_input.question)

            # Search for relevant travel data using cache
            items = search_data_with_cache(processed_question)

            if not items:
                return {"answer": "I'm sorry, I couldn't find any relevant travel information."}

            # Use the first item from the search results to generate a response
            response_data = items[0]
            prompt = f"User question: {processed_question}\nRelevant data: {response_data}\nGenerate a response."
            ai_response = await call_openai_api(prompt)

            return {"answer": ai_response}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# 他のサポートサービスも同様に実装

# Endpoint for Medical Diagnosis Support
@app.post("/medical_diagnosis")
async def medical_diagnosis(user_input: UserInput):
    try:
        # Validate the input
        validate_input(user_input.question)

        # Preprocess the input
        processed_question = preprocess_input(user_input.question)

        # Search for relevant medical data using cache
        items = search_data_with_cache(processed_question)

        if not items:
            return {"answer": "I'm sorry, I couldn't find any relevant medical information."}

        # Use the first item from the search results to generate a response
        response_data = items[0]
        prompt = f"User question: {processed_question}\nRelevant data: {response_data}\nGenerate a response."
        ai_response = await call_openai_api(prompt)

        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Academic Research Support
@app.post("/academic_research")
async def academic_research(user_input: UserInput):
    try:
        # Validate the input
        validate_input(user_input.question)

        # Preprocess the input
        processed_question = preprocess_input(user_input.question)

        # Search for relevant data using cache
        items = search_data_with_cache(processed_question)

        if not items:
            return {"answer": "I'm sorry, I couldn't find any relevant information for your academic research."}

        # Use the first item from the search results to generate a response
        response_data = items[0]
        prompt = f"User question: {processed_question}\nRelevant data: {response_data}\nGenerate a response."
        ai_response = await call_openai_api(prompt)

        return {"answer": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Coding Support
@app.post("/coding_support")
async def coding_support(user_input: UserInput):
    try:
        # Validate the input
        validate_input(user_input.question)

        # Preprocess the input
        processed_question = preprocess_input(user_input.question)

        # Generate a response using the LLM
        response = llm_chain.run(processed_question)

        # Return the response
        return {"answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class SupportService(ABC):
    @abstractmethod
    async def handle_request(self, user_input: UserInput):
        pass

@app.post("/legal_support")
async def legal_support(user_input: UserInput):
    service = LegalSupportService()
    return await service.handle_request(user_input)

@app.post("/travel_planning")
async def travel_planning(user_input: UserInput):
    service = TravelPlanningSupportService()
    return await service.handle_request(user_input)

# 他のエンドポイントも同様に更新 

# Function to extract article titles and links from HTML
# HTMLから記事タイトルとリンクを抽出する関数
def extract_articles_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    articles = []
    for link in soup.find_all('a', href=True):
        title = link.get_text()
        url = link['href']
        articles.append({'title': title, 'url': url})
    return articles

# Function to save JSON data to Azure Blob Storage
# JSONデータをAzure Blob Storageに保存する関数
def save_json_to_blob(data, container_name, blob_name):
    try:
        # Use DefaultAzureCredential to authenticate with Managed Identity
        # Managed Identityを使用して認証
        credential = DefaultAzureCredential()
        blob_service_client = azure_blob.BlobServiceClient(
            account_url=f"https://{os.getenv('AZURE_STORAGE_ACCOUNT_NAME')}.blob.core.windows.net",
            credential=credential
        )
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(json.dumps(data), overwrite=True)
        print(f"Data successfully saved to blob: {blob_name}")
    except Exception as e:
        print(f"Failed to save data to blob: {str(e)}")

# Example usage
# 例として、HTMLから記事を抽出し、Azure Blobに保存する
html_content = "<html><body><a href='https://example.com/article1'>Article 1</a><a href='https://example.com/article2'>Article 2</a></body></html>"
articles = extract_articles_from_html(html_content)
json_data = json.dumps(articles, ensure_ascii=False, indent=2)
save_json_to_blob(json_data, 'your-container-name', 'articles.json')

# Function to send a notification to Microsoft Teams
# Microsoft Teamsに通知を送信する関数
def send_teams_notification(webhook_url, message):
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "text": message
    }
    response = requests.post(webhook_url, headers=headers, json=payload)
    if response.status_code == 200:
        print("Notification sent to Teams successfully.")
    else:
        print(f"Failed to send notification to Teams: {response.status_code}")

# Example usage
# 例として、Teamsに通知を送信する
webhook_url = "https://outlook.office.com/webhook/your-webhook-url"
message = "A new blob has been uploaded."
send_teams_notification(webhook_url, message) 
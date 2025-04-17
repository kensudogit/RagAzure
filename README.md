# AI Chat System with Azure Services

## 概要
このプロジェクトは、Azureのサービスを利用してAIチャットシステムを構築するものです。システムは「検索」と「生成」の2つのフェーズに分かれており、ユーザの質問に対して適切な回答を生成します。

## 環境設定

### 必要なAzureサービス
- Azure OpenAI
- CosmosDB
- Azure Repos

### Python環境
- Python 3.x
- 必要なライブラリ: `azure-cosmos`, `azure-ai-openai`, `sentence-transformers`

## セットアップ手順

1. **Azureサービスの設定**
   - Azureポータルで必要なサービス（OpenAI, CosmosDB）を作成し、必要なAPIキーや接続情報を取得します。

2. **Python環境のセットアップ**
   - Pythonをインストールし、仮想環境を作成します。
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windowsの場合は `venv\Scripts\activate`
   ```
   - 必要なライブラリをインストールします。
   ```bash
   pip install azure-cosmos azure-ai-openai sentence-transformers
   ```

3. **ベクトル化とデータベースの準備**
   - `sentence-transformers`を使用してドキュメントをベクトル化し、CosmosDBに保存します。
   - ベクトル化の例:
   ```python
   from sentence_transformers import SentenceTransformer

   model = SentenceTransformer('all-MiniLM-L6-v2')
   documents = ["Document 1 text", "Document 2 text"]
   vectors = model.encode(documents)
   ```
   - CosmosDBへの保存例:
   ```python
   from azure.cosmos import CosmosClient

   cosmos_client = CosmosClient(os.getenv('COSMOS_ENDPOINT'), os.getenv('COSMOS_KEY'))
   database = cosmos_client.get_database_client('database_name')
   container = database.get_container_client('container_name')

   for doc, vector in zip(documents, vectors):
       container.upsert_item({
           'id': doc,
           'vector': vector.tolist()
       })
   ```

4. **AIチャットシステムの実装**
   - ユーザの質問を受け付け、CosmosDBをクエリして関連情報を検索します。
   - Azure OpenAIを使用してLLMにプロンプトを送信し、応答を受け取ります。
   - 応答をユーザに表示します。

5. **テストとデプロイ**
   - 様々なユーザ入力でシステムをテストし、期待通りに機能することを確認します。
   - 必要に応じてAzure Reposを使用してコードを管理し、デプロイします。

## 注意事項
- Azureサービスの利用には料金が発生する場合があります。事前に料金プランを確認してください。
- セキュリティ情報（APIキーなど）は安全に管理してください。

## ライセンス
このプロジェクトはMITライセンスの下で公開されています。 

from azure.ai.openai import OpenAIClient

openai_client = OpenAIClient(api_key="your-openai-api-key")
response = openai_client.completions.create(
    engine="davinci",
    prompt="Your prompt here",
    max_tokens=150
)
print(response.choices[0].text)

def search_cosmosdb(query_text):
    from sentence_transformers import SentenceTransformer
    from azure.cosmos import CosmosClient

    # モデルのロードと質問のベクトル化
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_vector = model.encode([query_text])[0]

    # CosmosDBクライアントの作成
    cosmos_client = CosmosClient(os.getenv('COSMOS_ENDPOINT'), os.getenv('COSMOS_KEY'))
    database = cosmos_client.get_database_client('your-database-name')
    container = database.get_container_client('your-container-name')

    # クエリを実行して関連情報を取得
    # ここでCosmosDBのベクトル類似性検索を実装
    # 例: SQLクエリを使用して類似性の高いドキュメントを取得
    query = "SELECT * FROM c WHERE ST_DISTANCE(c.vector, @query_vector) < @threshold"
    parameters = [
        {"name": "@query_vector", "value": query_vector.tolist()},
        {"name": "@threshold", "value": 0.5}  # 類似性の閾値を設定
    ]
    results = container.query_items(query=query, parameters=parameters, enable_cross_partition_query=True)

    return list(results)

def generate_response(prompt):
    from azure.ai.openai import OpenAIClient

    openai_client = OpenAIClient(api_key="your-openai-api-key")
    response = openai_client.completions.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text

def main():
    user_question = input("質問を入力してください: ")
    search_results = search_cosmosdb(user_question)
    
    # 検索結果をプロンプトに組み込む
    prompt = f"ユーザの質問: {user_question}\n関連情報: {search_results}\n回答を生成してください。"
    response = generate_response(prompt)
    
    print("AIの応答:", response)

if __name__ == "__main__":
    main()

## テストとデプロイ

### システムのテスト
- 様々なユーザ入力を使用して、システムが期待通りに動作するか確認します。
- 特に、以下の点を確認します:
  - ユーザの質問に対して適切な関連情報がCosmosDBから取得されるか。
  - Azure OpenAIを使用して生成された応答がユーザの質問に対して適切か。

### デプロイ
- 必要に応じて、Azure Reposを使用してコードを管理し、システムをデプロイします。
- デプロイの際には、以下の点に注意してください:
  - セキュリティ情報（APIキーなど）が安全に管理されていること。
  - Azureサービスの利用に伴う料金プランを確認し、予算内で運用できるようにすること。

### 追加の考慮事項
- **エラーハンドリング**: システムが予期しない入力やエラーに対して適切に対応できるように、エラーハンドリングを実装します。
- **パフォーマンス最適化**: ベクトル検索やLLMの応答生成のパフォーマンスを最適化し、ユーザ体験を向上させます。

### 次のステップ
- 上記のテストとデプロイの手順を実行し、システムが本番環境で安定して動作することを確認します。
- 必要に応じて、ユーザフィードバックをもとにシステムを改善します。

### ユニットテストの実装

まず、Pythonの`unittest`を使用して、主要な機能のユニットテストを作成します。

#### 例: `search_cosmosdb`関数のユニットテスト

```python
import unittest
from your_module import search_cosmosdb

class TestSearchCosmosDB(unittest.TestCase):
    def test_search_cosmosdb(self):
        # テスト用のクエリ
        query_text = "テストクエリ"
        
        # 関数の呼び出し
        results = search_cosmosdb(query_text)
        
        # 結果の検証
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)  # 結果が1件以上であることを確認

if __name__ == '__main__':
    unittest.main()
```

### Azureへのデプロイ

次に、Azure App Serviceを使用してアプリケーションをデプロイします。

#### 1. Azure CLIを使用したデプロイ

Azure CLIを使用して、アプリケーションをAzure App Serviceにデプロイします。

```bash
# Azureにログイン
az login

# リソースグループの作成
az group create --name myResourceGroup --location eastus

# App Serviceプランの作成
az appservice plan create --name myAppServicePlan --resource-group myResourceGroup --sku FREE

# Webアプリの作成
az webapp create --resource-group myResourceGroup --plan myAppServicePlan --name myUniqueAppName --runtime "PYTHON|3.8"

# コードのデプロイ
az webapp up --name myUniqueAppName --resource-group myResourceGroup --plan myAppServicePlan
```

#### 2. デプロイ後の確認

- AzureポータルでWebアプリの状態を確認し、正常に動作していることを確認します。
- 必要に応じて、Azure MonitorやApplication Insightsを設定し、アプリケーションのパフォーマンスを監視します。

### 次のステップ

- ユニットテストを実行し、すべてのテストがパスすることを確認します。
- Azureへのデプロイを完了し、アプリケーションが期待通りに動作することを確認します。
- 必要に応じて、ユーザフィードバックをもとにシステムを改善します。

これらの手順を進める中で、さらなるサポートが必要な場合はお知らせください。 

python -m unittest discover -s tests 

pytest tests/ 

git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin master 

# FastAPI AI Chat System

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd workspace/RagAzure
   ```

2. **Set Up Python Environment**
   - Create and activate a virtual environment (optional but recommended):
     ```bash
     # On Windows
     python -m venv venv
     .\venv\Scripts\activate

     # On macOS/Linux
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Required Packages**
   ```bash
   pip install fastapi uvicorn azure-cosmos
   ```

4. **Set Environment Variables**
   - Ensure you have the following environment variables set for CosmosDB:
     ```bash
     export COSMOS_ENDPOINT="<your-cosmos-endpoint>"
     export COSMOS_KEY="<your-cosmos-key>"
     ```

## Running the Application

1. **Start the FastAPI Server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Access the Application**
   - Open your browser and go to `http://localhost:8000` to access the application.

## Additional Information

- Ensure your IDE is configured to use the correct Python interpreter where the packages are installed.
- If you encounter any issues, verify that all dependencies are installed and environment variables are correctly set. 

# Windowsの場合
set COSMOS_ENDPOINT=<your-cosmos-endpoint>
set COSMOS_KEY=<your-cosmos-key>

# macOS/Linuxの場合
export COSMOS_ENDPOINT=<your-cosmos-endpoint>
export COSMOS_KEY=<your-cosmos-key>

## データ検索の実装

### 効率的なデータ検索
既存データを検索するために、Azure Cosmos DBを使用します。以下の手順でデータを効率的に検索します。

1. **インデックスの利用**: データベースのインデックスを適切に設定し、検索速度を向上させます。
2. **キャッシュの利用**: `cachetools`を使用して、検索結果をキャッシュし、データベースへのアクセス回数を減らします。

#### サンプルコード
```python
from cachetools import cached, TTLCache

# キャッシュを設定（TTL: 10分）
cache = TTLCache(maxsize=100, ttl=600)

@cached(cache)
def search_data_with_cache(question):
    return search_data_in_cosmosdb(question)
```

### セキュリティ要件
データ検索におけるセキュリティを強化するために、以下の点を考慮します。

1. **入力のバリデーション**: ユーザーからの入力を適切にバリデーションし、SQLインジェクションなどの攻撃を防ぎます。
2. **環境変数の利用**: APIキーやデータベースの接続情報は環境変数を使用して管理します。

#### サンプルコード
```python
from fastapi import HTTPException

def validate_input(question):
    if not question or len(question) > 256:
        raise HTTPException(status_code=400, detail="Invalid input")
```

これらの手法を用いることで、効率的かつ安全にデータを検索することができます。 "# RagAzure" 

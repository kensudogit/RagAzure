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

    # 応答の精度を評価
    # テスト質問に対する応答が期待通りかを確認
    for test in TEST_QUESTIONS:
        if user_question == test["question"]:
            if not evaluate_response_accuracy(response, test["expected_answer"]):
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
     export OPENAI_API_KEY="<your-openai-api-key>"
     export AZURE_STORAGE_ACCOUNT_NAME="<your-storage-account-name>"
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
set OPENAI_API_KEY=<your-openai-api-key>
set AZURE_STORAGE_ACCOUNT_NAME=<your-storage-account-name>

# macOS/Linuxの場合
export COSMOS_ENDPOINT=<your-cosmos-endpoint>
export COSMOS_KEY=<your-cosmos-key>
export OPENAI_API_KEY=<your-openai-api-key>
export AZURE_STORAGE_ACCOUNT_NAME=<your-storage-account-name>

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

これらの手法を用いることで、効率的かつ安全にデータを検索することができます。

## LLM Switching Feature

This project includes a feature that automatically switches between different Language Learning Models (LLMs) based on the accuracy of their responses.

### Configuration

- Set the environment variable `LLM_CHOICE` to choose the default LLM. Options are `gpt4all` or `openai`.
  ```bash
  export LLM_CHOICE=gpt4all  # or 'openai'
  ```

### Testing the LLM Switching

1. **Run the Application**: Start the application using the command:
   ```bash
   uvicorn app.main:app --reload
   ```

2. **Test Questions**: Use predefined test questions to evaluate the accuracy of the LLM responses. If the accuracy is low, the system will automatically switch to another LLM.

3. **Monitor Logs**: Check the application logs to verify that the LLM switching occurs as expected when the accuracy threshold is not met.

4. **Adjust Thresholds**: Modify the accuracy evaluation logic in `main.py` if needed to better suit your use case.

## ビルド手順

1. **リポジトリのクローン**
   - このプロジェクトのリポジトリをローカルマシンにクローンします。
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Python環境のセットアップ**
   - 仮想環境を作成し、必要なパッケージをインストールします。
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windowsの場合は `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **環境変数の設定**
   - 必要なAzureサービスの接続情報やAPIキーを環境変数として設定します。
   ```bash
   export COSMOS_ENDPOINT=<your-cosmos-endpoint>
   export COSMOS_KEY=<your-cosmos-key>
   export OPENAI_API_KEY=<your-openai-api-key>
   export AZURE_STORAGE_ACCOUNT_NAME=<your-storage-account-name>
   ```

4. **アプリケーションの起動**
   - FastAPIアプリケーションを起動します。
   ```bash
   uvicorn app.main:app --reload
   ```

5. **動作確認**
   - ブラウザで `http://localhost:8000` にアクセスし、アプリケーションが正常に動作していることを確認します。

## スクレイピングとアップロードスクリプトのセットアップ

このセクションでは、ウェブサイトからデータをスクレイピングし、Azure Blob Storageにアップロードするスクリプトのセットアップ手順を説明します。

1. **Azure Blob Storageの設定**
   - AzureポータルでBlob Storageアカウントを作成し、接続文字列を取得します。
   - コンテナを作成し、データを保存する準備をします。

2. **Python環境のセットアップ**
   - Pythonをインストールし、仮想環境を作成します。
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windowsの場合は `venv\Scripts\activate`
   ```
   - 必要なライブラリをインストールします。
   ```bash
   pip install requests beautifulsoup4 azure-storage-blob
   ```

3. **スクリプトの実行**
   - `crape_and_upload.py` または `scrape_to_md_json_and_upload.py` を実行して、データをスクレイピングし、Azure Blob Storageにアップロードします。
   ```bash
   python crape_and_upload.py
   ```
   または
   ```bash
   python scrape_to_md_json_and_upload.py
   ```

4. **結果の確認**
   - AzureポータルでBlob Storageを確認し、データが正しくアップロードされていることを確認します。

from llamacpp import LlamaIndex

# データセットのロード
dataset_path = 'path/to/your/dataset'
index = LlamaIndex(dataset_path)
index.build()

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

## インフラストラクチャをコードとして管理（IaC）とCI/CDの整備

### インフラストラクチャをコードとして管理（IaC）

1. **IaCツールの選択:**
   - 人気のある選択肢には、Terraform、AWS CloudFormation、Azure Resource Manager (ARM) テンプレートがあります。このガイドでは、Terraformを例に使用します。

2. **Terraformのインストール:**
   - [公式サイト](https://www.terraform.io/downloads.html)からTerraformをダウンロードしてインストールします。
   - ターミナルで`terraform --version`を実行してインストールを確認します。

3. **Terraformの設定:**
   - Terraformの設定ファイル用のディレクトリを作成します。
   - `.tf`ファイルにインフラストラクチャを定義します。例えば、Azureのリソースグループを作成するには以下のように記述します。
     ```hcl
     provider "azurerm" {
       features {}
     }

     resource "azurerm_resource_group" "example" {
       name     = "example-resources"
       location = "East US"
     }
     ```

4. **Terraformの初期化:**
   - `terraform init`を実行してディレクトリを初期化し、必要なプラグインをダウンロードします。

5. **プランと適用:**
   - `terraform plan`を使用して、どのような変更が行われるかを確認します。
   - `terraform apply`で設定を適用します。

6. **バージョン管理:**
   - TerraformファイルをGitのようなバージョン管理システムに保存します。

### CI/CDパイプラインの整備

1. **CI/CDツールの選択:**
   - Jenkins、GitHub Actions、GitLab CI/CD、Azure DevOpsなどのオプションがあります。この例ではGitHub Actionsを使用します。

2. **GitHubリポジトリの作成:**
   - コードベースとTerraformファイルをGitHubリポジトリにプッシュします。

3. **GitHub Actionsの設定:**
   - リポジトリ内に`.github/workflows`ディレクトリを作成します。
   - CI/CDパイプライン用のYAMLファイル（例: `ci-cd.yml`）を追加します。
     ```yaml
     name: CI/CD Pipeline

     on:
       push:
         branches:
           - main

     jobs:
       build:
         runs-on: ubuntu-latest

         steps:
         - name: Checkout code
           uses: actions/checkout@v2

         - name: Set up Terraform
           uses: hashicorp/setup-terraform@v1
           with:
             terraform_version: 1.0.0

         - name: Terraform Init
           run: terraform init

         - name: Terraform Plan
           run: terraform plan

         - name: Terraform Apply
           run: terraform apply -auto-approve
           env:
             ARM_CLIENT_ID: ${{ secrets.ARM_CLIENT_ID }}
             ARM_CLIENT_SECRET: ${{ secrets.ARM_CLIENT_SECRET }}
             ARM_SUBSCRIPTION_ID: ${{ secrets.ARM_SUBSCRIPTION_ID }}
             ARM_TENANT_ID: ${{ secrets.ARM_TENANT_ID }}
     ```

4. **シークレットの設定:**
   - GitHubリポジトリの設定で、Settings > Secretsに移動し、認証に必要なシークレット（例: Azureの資格情報）を追加します。

5. **パイプラインのテスト:**
   - `main`ブランチに変更をプッシュしてパイプラインをトリガーします。
   - GitHubのActionsタブでパイプラインが正常に実行されることを確認します。

6. **デプロイと監視:**
   - パイプラインが設定されると、リポジトリに更新をプッシュするたびに自動的にインフラストラクチャがデプロイされます。

### 追加の考慮事項

- **セキュリティ:** APIキーや資格情報などの機密情報は、環境変数やシークレット管理ツールを使用して安全に管理します。
- **テスト:** インフラストラクチャとアプリケーションコードを検証するために、ユニットテストと統合テストを実装します。
- **監視:** デプロイされたアプリケーションとインフラストラクチャのパフォーマンスと状態を監視するためのツールを使用します。

これらの手順に従うことで、インフラストラクチャとアプリケーションのデプロイと管理を自動化する信頼性の高いIaCとCI/CDのセットアップを確立できます。

## Azure DevOps Servicesの導入・設定手順

### 1. Azure DevOpsアカウントの作成
- [Azure DevOps](https://dev.azure.com/)にアクセスし、Microsoftアカウントでサインインします。
- 必要に応じて新しいアカウントを作成します。

### 2. プロジェクトの作成
- Azure DevOpsポータルで「新しいプロジェクト」をクリックします。
- プロジェクト名と説明を入力し、プロジェクトの可視性（パブリックまたはプライベート）を選択します。
- 「作成」をクリックしてプロジェクトを作成します。

### 3. リポジトリの設定
- プロジェクト内で「リポジトリ」タブを選択します。
- 「新しいリポジトリ」をクリックし、リポジトリ名を入力します。
- 必要に応じて、リポジトリの初期化オプションを選択します（例: READMEファイルの追加）。

### 4. パイプラインの設定
- 「パイプライン」タブを選択し、「新しいパイプライン」をクリックします。
- コードの場所を選択し、リポジトリを選択します。
- ビルドパイプラインの設定を行い、必要なタスクを追加します。
- パイプラインを保存して実行します。

### 5. Azure Boardsの使用
- 「Boards」タブを選択し、作業項目を作成します。
- バックログやスプリントを設定し、チームの作業を管理します。

### 6. Azure Artifactsの設定
- 「Artifacts」タブを選択し、新しいフィードを作成します。
- パッケージをフィードに公開し、プロジェクト内で共有します。

### 7. セキュリティとアクセス管理
- プロジェクト設定で「セキュリティ」オプションを選択します。
- ユーザーやグループのアクセス権を設定し、プロジェクトのセキュリティを管理します。

これらの手順を通じて、Azure DevOps Servicesを使用してプロジェクトの管理と継続的インテグレーション/デリバリーを効率化できます。

# LLMの選択に基づいて初期化
# デフォルトは 'gpt4all'
llm_choice = os.getenv('LLM_CHOICE', 'gpt4all')

# ヘルスチェックエンドポイント
# サーバーが正常に動作しているかを確認するためのエンドポイント
@app.get("/")
async def root():
    return {"message": "AI Chat System is running"}

# HTMLから記事タイトルとリンクを抽出する関数
# BeautifulSoupを使用してHTMLコンテンツを解析し、記事のタイトルとリンクを抽出
def extract_articles_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    articles = []
    for link in soup.find_all('a', href=True):
        title = link.get_text()
        url = link['href']
        articles.append({'title': title, 'url': url})
    return articles

# JSONデータをAzure Blob Storageに保存する関数
# AzureのBlob Storageにデータを保存するための関数
def save_json_to_blob(data, container_name, blob_name):
    try:
        # Use DefaultAzureCredential to authenticate with Managed Identity
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

# Microsoft Teamsに通知を送信する関数
# 指定されたWebhook URLを使用してTeamsにメッセージを送信
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

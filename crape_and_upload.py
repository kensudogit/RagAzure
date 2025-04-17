import requests
from bs4 import BeautifulSoup
from azure.storage.blob import BlobServiceClient, ContentSettings

# このスクリプトはウェブサイトからHTMLをスクレイピングし、Azure Blob Storageにアップロードするためのものです。

# ========== 設定 ==========
# URL: スクレイピング対象のウェブサイトのURL
# AZURE_CONNECTION_STRING: Azure Blob Storageへの接続文字列
# CONTAINER_NAME: データを保存するコンテナの名前
# BLOB_NAME: アップロードするファイルの名前
# ==========================

# scrape_website関数は指定されたURLからHTMLを取得し、整形して返します。
# url: スクレイピング対象のウェブサイトのURL
# 戻り値: 整形されたHTMLコンテンツ

def scrape_website(url):
    response = requests.get(url)
    response.raise_for_status()  # エラー時に例外を発生させる
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.prettify()

# upload_to_azure_blob関数はHTMLデータをAzure Blob Storageにアップロードします。
# html_data: アップロードするHTMLデータ
# connection_string: Azure Blob Storageへの接続文字列
# container_name: データを保存するコンテナの名前
# blob_name: アップロードするファイルの名前

def upload_to_azure_blob(html_data, connection_string, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)

    # コンテナが存在しなければ作成
    try:
        container_client.create_container()
    except Exception:
        pass  # 既に存在する場合は無視

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        html_data,
        overwrite=True,
        content_settings=ContentSettings(content_type='text/html')
    )
    print(f"アップロード完了: https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}")

# main関数はスクリプトのエントリーポイントです。
# ウェブサイトからHTMLを取得し、Azure Blob Storageにアップロードします。

def main():
    print(f"{URL} からHTMLを取得中...")
    html_content = scrape_website(URL)
    print("Azure Blobへアップロード中...")
    upload_to_azure_blob(html_content, AZURE_CONNECTION_STRING, CONTAINER_NAME, BLOB_NAME)

if __name__ == "__main__":
    main()

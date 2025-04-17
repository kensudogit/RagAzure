import requests
from bs4 import BeautifulSoup
from azure.storage.blob import BlobServiceClient, ContentSettings
import json
from azure.core.exceptions import ResourceExistsError

# ========== 設定 ==========
URL = "https://example.com/news"  # ニュース一覧ページなど
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=<your_account>;AccountKey=<your_key>;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "scrape-output"
BLOB_NAME_MD = "articles.md"
BLOB_NAME_JSON = "articles.json"
# ==========================

def scrape_articles(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # ★ 適宜調整（例：記事は <h2><a href=...>タイトル</a></h2>）
    articles = []
    for h2 in soup.find_all("h2"):
        a = h2.find("a")
        if a and a.get("href"):
            title = a.get_text(strip=True)
            link = a["href"]
            if not link.startswith("http"):
                link = requests.compat.urljoin(url, link)
            articles.append({"title": title, "url": link})
    return articles

def convert_to_markdown(articles):
    return "\n".join([f"- [{a['title']}]({a['url']})" for a in articles])

def upload_blob(data, filename, content_type, conn_str, container):
    blob_service = BlobServiceClient.from_connection_string(conn_str)
    container_client = blob_service.get_container_client(container)
    try:
        container_client.create_container()
    except ResourceExistsError:
        pass  # 既存でもOK

    blob_client = container_client.get_blob_client(filename)
    blob_client.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type)
    )
    print(f"アップロード完了: https://{blob_service.account_name}.blob.core.windows.net/{container}/{filename}")

def main():
    print("記事を収集中...")
    articles = scrape_articles(URL)

    print("Markdown変換中...")
    md_data = convert_to_markdown(articles)

    print("Azure Blobへアップロード中...")
    upload_blob(md_data, BLOB_NAME_MD, 'text/markdown', AZURE_CONNECTION_STRING, CONTAINER_NAME)

    print("JSON変換・アップロード中...")
    json_data = json.dumps(articles, indent=2, ensure_ascii=False)
    upload_blob(json_data, BLOB_NAME_JSON, 'application/json', AZURE_CONNECTION_STRING, CONTAINER_NAME)

if __name__ == "__main__":
    main()

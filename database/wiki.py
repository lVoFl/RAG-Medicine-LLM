"""
Wikipedia 糖尿病相关文章爬虫
使用 MediaWiki API 爬取中文/英文维基百科中糖尿病相关内容
"""

import requests
import json
import time
import os

# ─── 配置 ────────────────────────────────────────────────────────────────────

WIKIS = {
    "zh": "https://zh.wikipedia.org/w/api.php",   # 中文维基
    "en": "https://en.wikipedia.org/w/api.php",   # 英文维基
}

# 爬取的语言（改成 "en" 只爬英文，或保留两者）
LANGUAGES = ["zh", "en"]

# 糖尿病相关搜索关键词
SEARCH_QUERIES = {
    "zh": [
        "糖尿病", "1型糖尿病", "2型糖尿病", "妊娠糖尿病",
        "胰岛素", "胰岛素抵抗", "血糖", "高血糖",
        "糖化血红蛋白", "低血糖症", "糖尿病并发症",
        "糖尿病视网膜病变", "糖尿病肾病", "糖尿病神经病变",
        "二甲双胍", "胰岛素泵", "连续血糖监测",
    ],
    "en": [
        "Diabetes mellitus", "Type 1 diabetes", "Type 2 diabetes",
        "Gestational diabetes", "Insulin", "Insulin resistance",
        "Blood sugar", "Hyperglycemia", "Hypoglycemia",
        "Glycated hemoglobin", "Diabetic retinopathy",
        "Diabetic nephropathy", "Diabetic neuropathy",
        "Metformin", "Insulin pump", "Continuous glucose monitor",
        "Diabetic ketoacidosis", "Pancreatic islets",
    ],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "wiki_diabetes")
REQUEST_DELAY = 1.0   # 每次请求间隔（秒），遵守礼貌爬取原则
MAX_RESULTS_PER_QUERY = 5  # 每个关键词最多取几篇文章

HEADERS = {
    "User-Agent": "DiabetesResearchBot/1.0 (graduation project; educational use)"
}

# ─── MediaWiki API 工具函数 ───────────────────────────────────────────────────

def search_articles(api_url: str, query: str, limit: int = MAX_RESULTS_PER_QUERY) -> list[dict]:
    """搜索与关键词相关的文章列表"""
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "srinfo": "totalhits",
        "srprop": "snippet|titlesnippet",
        "format": "json",
    }
    resp = requests.get(api_url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("query", {}).get("search", [])


def fetch_article(api_url: str, title: str) -> dict | None:
    """获取单篇文章的详细内容（摘要 + 正文各节 + 分类 + 链接）"""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|categories|links|info|pageprops",
        "exintro": False,       # False = 获取全文，True = 只要导言
        "explaintext": True,    # 纯文本（去除 Wiki 标记）
        "cllimit": 50,
        "pllimit": 100,
        "inprop": "url",
        "format": "json",
        "redirects": True,
    }
    resp = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    page = next(iter(pages.values()))

    if "missing" in page:
        return None

    categories = [c["title"] for c in page.get("categories", [])]
    links = [l["title"] for l in page.get("links", [])]

    # 将正文按段落分割，过滤空行
    raw_text: str = page.get("extract", "") or ""
    paragraphs = [p.strip() for p in raw_text.split("\n") if p.strip()]

    return {
        "pageid": page.get("pageid"),
        "title": page.get("title"),
        "url": page.get("fullurl", ""),
        "categories": categories,
        "links": links[:50],           # 只保留前 50 条链接，避免数据过大
        "text": raw_text,
        "paragraphs": paragraphs,
        "word_count": len(raw_text),
    }


def fetch_summary(api_url: str, title: str) -> str:
    """使用 REST API 获取简洁摘要（可选，英文维基效果更好）"""
    # REST summary endpoint
    base = api_url.replace("/w/api.php", "")
    url = f"{base}/api/rest_v1/page/summary/{requests.utils.quote(title)}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("extract", "")
    except Exception:
        pass
    return ""


# ─── 主爬取逻辑 ───────────────────────────────────────────────────────────────

def scrape_language(lang: str, api_url: str, queries: list[str]) -> list[dict]:
    """爬取指定语言维基百科"""
    seen_ids: set[int] = set()
    articles: list[dict] = []

    print(f"\n{'='*60}")
    print(f"  开始爬取 [{lang.upper()}] 维基百科")
    print(f"{'='*60}")

    for query in queries:
        print(f"\n[搜索] {query}")
        try:
            results = search_articles(api_url, query)
        except Exception as e:
            print(f"  搜索失败: {e}")
            time.sleep(REQUEST_DELAY)
            continue

        for hit in results:
            title = hit["title"]
            pageid = hit.get("pageid", -1)

            if pageid in seen_ids:
                print(f"  跳过 (已爬): {title}")
                continue
            seen_ids.add(pageid)

            print(f"  获取文章: {title} ...", end=" ", flush=True)
            try:
                article = fetch_article(api_url, title)
                if article is None:
                    print("未找到")
                    continue

                article["lang"] = lang
                article["query"] = query
                articles.append(article)
                print(f"OK  ({article['word_count']:,} 字符)")
            except Exception as e:
                print(f"失败 ({e})")

            time.sleep(REQUEST_DELAY)

    return articles


def save_results(articles: list[dict], lang: str):
    """将结果保存为 JSON 及纯文本两种格式"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 完整 JSON ──
    json_path = os.path.join(OUTPUT_DIR, f"diabetes_{lang}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"  [保存] JSON -> {json_path}")

    # ── 纯文本（便于人工阅读或 NLP 处理）──
    txt_path = os.path.join(OUTPUT_DIR, f"diabetes_{lang}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(f"{'#'*80}\n")
            f.write(f"标题: {art['title']}\n")
            f.write(f"URL:  {art['url']}\n")
            f.write(f"分类: {', '.join(art['categories'][:10])}\n")
            f.write(f"{'─'*80}\n")
            f.write(art["text"])
            f.write("\n\n")
    print(f"  [保存] TXT  -> {txt_path}")

    # ── 摘要表格（CSV）──
    csv_path = os.path.join(OUTPUT_DIR, f"diabetes_{lang}_index.csv")
    with open(csv_path, "w", encoding="utf-8-sig") as f:
        f.write("pageid,title,url,word_count,categories\n")
        for art in articles:
            cats = "|".join(art["categories"][:5]).replace(",", "；")
            title_esc = art["title"].replace(",", "，")
            f.write(f"{art['pageid']},{title_esc},{art['url']},{art['word_count']},{cats}\n")
    print(f"  [保存] CSV  -> {csv_path}")


def print_stats(articles: list[dict], lang: str):
    total_chars = sum(a["word_count"] for a in articles)
    print(f"\n[{lang.upper()}] 统计: 共 {len(articles)} 篇文章, "
          f"合计 {total_chars:,} 字符")


# ─── 入口 ─────────────────────────────────────────────────────────────────────

def main():
    all_articles: dict[str, list[dict]] = {}

    for lang in LANGUAGES:
        if lang not in SEARCH_QUERIES:
            continue
        api_url = WIKIS[lang]
        queries = SEARCH_QUERIES[lang]

        articles = scrape_language(lang, api_url, queries)
        all_articles[lang] = articles

        save_results(articles, lang)
        print_stats(articles, lang)

    # 合并所有语言到一个文件（可选）
    merged = [a for arts in all_articles.values() for a in arts]
    merged_path = os.path.join(OUTPUT_DIR, "diabetes_all.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n[完成] 合并文件 -> {merged_path}")
    print(f"[完成] 共爬取 {len(merged)} 篇文章")


if __name__ == "__main__":
    main()

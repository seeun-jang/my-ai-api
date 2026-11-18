"""Public bibliographic metadata. No invented titles, full-text harvesting or credentials."""
import html
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlencode

import requests

AGENT = "PRISM-AI-Education/1.0 (+https://github.com/seeun-jang/prism-ai-app)"
_cache = {}
_lock = threading.Lock()
_ol_lock = threading.Lock()
_last_ol = 0.0


def clean(value, limit=600):
    return html.unescape(re.sub(r"<[^>]*>", "", str(value or ""))).strip()[:limit]


def fetch(provider, query):
    global _last_ol
    cache_key = (provider, query.casefold())
    with _lock:
        cached = _cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < 600:
            return cached[1]
    if provider == "books":
        url = "https://openlibrary.org/search.json"
        params = {"q": query, "limit": 6,
                  "fields": "key,title,author_name,first_publish_year,cover_i,isbn"}
        # Open Library's anonymous limit is one request per second.
        with _ol_lock:
            remaining = 1.1 - (time.monotonic() - _last_ol)
            if remaining > 0:
                time.sleep(remaining)
            _last_ol = time.monotonic()
            response = requests.get(url, params=params, headers={"User-Agent": AGENT}, timeout=(4, 10))
    else:
        url = "https://api.crossref.org/works"
        params = {"query": query, "rows": 6, "filter": "type:journal-article",
                  "select": "DOI,title,author,published,container-title,type"}
        response = requests.get(url, params=params, headers={"User-Agent": AGENT}, timeout=(4, 10))
    response.raise_for_status()
    data = response.json()
    results, seen = [], set()
    if provider == "books":
        docs = data.get("docs", [])
        for item in docs[:10]:
            key = item.get("key", "")
            title = clean(item.get("title"), 350)
            if not re.fullmatch(r"/works/OL\d+W", key) or not title or key in seen:
                continue
            seen.add(key)
            cover = item.get("cover_i")
            results.append({"title": title,
                "authors": [clean(a, 100) for a in item.get("author_name", [])[:4]],
                "year": item.get("first_publish_year") if isinstance(item.get("first_publish_year"), int) else None,
                "url": "https://openlibrary.org" + key, "source": "Open Library",
                "cover": f"https://covers.openlibrary.org/b/id/{cover}-M.jpg?default=false" if isinstance(cover, int) and cover > 0 else None})
    else:
        for item in data.get("message", {}).get("items", [])[:10]:
            doi = item.get("DOI", "")
            titles = item.get("title") or []
            if not re.fullmatch(r"10\.\d{4,9}/\S+", doi) or not titles or doi in seen:
                continue
            seen.add(doi)
            parts = item.get("published", {}).get("date-parts", [[]])
            authors = [clean(" ".join([a.get("given", ""), a.get("family", "")]), 100)
                       for a in item.get("author", [])[:4]]
            results.append({"title": clean(titles[0], 350), "authors": authors,
                "year": parts[0][0] if parts and parts[0] and isinstance(parts[0][0], int) else None,
                "url": "https://doi.org/" + quote(doi, safe="/"), "source": "Crossref",
                "journal": clean((item.get("container-title") or [""])[0], 200)})
    with _lock:
        if len(_cache) >= 128:
            _cache.pop(next(iter(_cache)))
        _cache[cache_key] = (time.monotonic(), results)
    return results


def discover(query, kind="all"):
    query = query.strip()[:160]
    providers = ["books", "papers"] if kind == "all" else [kind]
    result = {"query": query, "books": [], "papers": [], "warnings": []}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(fetch, name, query) for name in providers}
        for name, future in futures.items():
            try:
                result[name] = future.result()
            except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
                label = "도서" if name == "books" else "논문"
                result["warnings"].append(f"{label} 검색 서버가 응답하지 않았습니다. 아래 외부 검색을 이용하거나 잠시 뒤 다시 검색해 주세요.")
    result["external_links"] = {
        "교보문고에서 검색": "https://search.kyobobook.co.kr/search?" + urlencode({"keyword": query}),
        "arXiv에서 검색": "https://arxiv.org/search/?" + urlencode({"query": query, "searchtype": "all"}),
    }
    return result

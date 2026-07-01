import argparse, json, re, sys, time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
LISTING_BASE = "https://www.shl.com/solutions/products/product-catalog"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"
PAGE_SIZE = 12
MAX_PAGES = 40
REQUEST_DELAY = 0.6
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
BOILERPLATE_MARKERS = [
    "Client Support", "Contact Support", "Practice Tests", "Browser Check",
    "Global Offices", "Speak to our team", "Support chat", "Back to Product Catalog",
]
TEST_TYPE_LETTERS = {"A", "B", "C", "D", "E", "K", "P", "S"}
DURATION_PATTERNS = [
    re.compile(r"Approximate Completion Time[:\s]*([0-9]+)\s*min", re.I),
    re.compile(r"is\s+([0-9]+)\s+minutes?\s+long", re.I),
    re.compile(r"([0-9]+)\s*minutes?\s+(?:test|assessment)", re.I),
]
JOB_LEVEL_KEYWORDS = [
    "Graduate", "Entry-Level", "Mid-Professional", "Professional Individual Contributor",
    "Supervisor", "Manager", "Director", "Executive", "General Population", "All Levels",
]
session = requests.Session()
session.headers.update(HEADERS)
def _get(url):
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code != 200:
            print(f" [!] {resp.status_code} for {url}")
            return None
        return resp
    except requests.RequestException as e:
        print(f" [!] request failed for {url}: {e}")
    return None
def fetch_listing_pages(type_filter="1"):
    """Paginate the catalog index filtered to Individual Test Solutions
    (type=1, based on observed query params). If this returns nothing, the
    listing is likely a client-side XHR call — see notes at bottom of file.
    """
    seen, urls = set(), []
    for page_num in range(MAX_PAGES):
        start = page_num * PAGE_SIZE
        params = {"start": start, "type": type_filter, "f": 1}
        resp = session.get(LISTING_BASE, params=params, timeout=20, headers=HEADERS)
        if resp.status_code != 200:
            print(f" [!] listing page {page_num} -> {resp.status_code}, stopping")
            break
        soup = BeautifulSoup(resp.text, "html.parser")  
        page_links = _extract_product_links(soup)
        new_links = [u for u in page_links if u not in seen]
        if not new_links:
            print(f" [i] no new links at start={start}, stopping pagination")
            break
        for u in new_links:
            seen.add(u); urls.append(u)
        print(f" [i] page {page_num}: +{len(new_links)} links (total {len(urls)})")
        time.sleep(REQUEST_DELAY)
    return urls
def _extract_product_links(soup):
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/product-catalog/view/" in href:
            full = href if href.startswith("http") else f"https://www.shl.com{href}"
            links.append(full.split("?")[0])
    return sorted(set(links))
def parse_product_page(url):
    resp = _get(url)
    if resp is None:
        raise RuntimeError(f"could not fetch {url}")
    soup = BeautifulSoup(resp.text, "html.parser")
    full_text = soup.get_text(separator="\n")
    lines = [l.strip() for l in full_text.split("\n") if l.strip()]
    name = _extract_name(soup, url)
    return {
        "id": url.rstrip("/").split("/")[-1],
        "name": name,
        "url": url,
        "test_type": _extract_test_type(full_text),
        "description": _extract_description(lines, name),
        "duration_minutes": _extract_duration(full_text),
        "job_level": _extract_job_levels(full_text),
    }
def _extract_name(soup, url):
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    if soup.title and soup.title.string:
        return soup.title.string.split("|")[0].strip()
    return url.rstrip("/").split("/")[-1].replace("-", " ").title()
def _extract_description(lines, name):
    # SHL pages repeat description right after the name, e.g.
    # "Java 8 (New): Multi-choice test that measures..."
    for line in lines:
        if line.startswith(name) and ":" in line:
            candidate = line.split(":", 1)[1].strip()
            if len(candidate) > 20:
                return candidate
    for line in lines:
        if len(line) < 40 or line == name:
            continue
        if any(m in line for m in BOILERPLATE_MARKERS):
            continue
        return line
    return None
def _extract_test_type(full_text):
    m = re.search(r"\[([A-Z])\]", full_text)
    if m and m.group(1) in TEST_TYPE_LETTERS:
        return m.group(1)
    return None
def _extract_duration(full_text):
    for pattern in DURATION_PATTERNS:
        m = pattern.search(full_text)
        if m:
            return int(m.group(1))
    return None
def _extract_job_levels(full_text):
    return [kw for kw in JOB_LEVEL_KEYWORDS if kw in full_text]
def build_catalog(urls=None):
    if urls is None:
        print("[i] fetching listing pages...")
        urls = fetch_listing_pages()
        print(f"[i] found {len(urls)} product URLs")
    if not urls:
        print("[!] no URLs found — see 'IF THIS RETURNS NOTHING' note.")
        return
    records = []
    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            rec = parse_product_page(url)
            if not rec["name"] or not rec["description"]:
                print(f" [!] incomplete, check selectors: {rec}")
            records.append(rec)
        except Exception as e:
            print(f" [!] skip {url}: {e}")
            time.sleep(REQUEST_DELAY)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[i] wrote {len(records)} records to {OUTPUT_PATH}")
def debug_one(url):
    print(json.dumps(parse_product_page(url), indent=2))
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-one", metavar="URL")
    parser.add_argument("--urls-file", metavar="PATH")
    args = parser.parse_args()
    if args.debug_one:
        debug_one(args.debug_one); sys.exit(0)
    if args.urls_file:
        urls = [l.strip() for l in Path(args.urls_file).read_text().splitlines() if l.strip()]
        build_catalog(urls)
    else:
        build_catalog()
# IF fetch_listing_pages() RETURNS NOTHING: the listing is likely populated
# via a client-side XHR call, not server-rendered HTML. Open the catalog
# page in Chrome DevTools -> Network -> Fetch/XHR, click to page 2, find the
# JSON request, and hit that URL directly instead of parsing HTML.

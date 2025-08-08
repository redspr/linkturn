import re
import sys
import requests
import pickle
import os
import numpy as np
from bs4 import BeautifulSoup
from urllib.parse import urljoin

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


EXCLUDE_KEYWORDS = [
    "jquery", "bootstrap", "react", "vue", "angular", "svelte", "ember", "backbone", "lodash", "underscore", "moment", "gtag", 
    "ga.js", "analytics", "tagmanager", "googletag", "google-analytics", "mixpanel", "hotjar", "matomo", "segment", "clarity", 
    "facebook", "fbq", "pixel", "snowplow", "ads", "adservice", "doubleclick", "googlesyndication", "adtrack", "tracking", 
    "bidder", "criteo", "taboola", "outbrain", "cdn", "cloudflare", "jsdelivr", "unpkg", "akamai", "netdna", "maxcdn", "fastly",
    "fontawesome", "material", "uikit", "zurb", "foundation", "modernizr", "polyfill", "babel", "webpack", "require", "systemjs", 
    "rollup", "esbuild", "tslib", "min.js", "external", "thirdparty", "third-party", "common.js", "gsi", "gtm", "datatable"
]

CNN_MODEL_PATH = "lib/models/cnndata.h5"
CNN_TOKEN_PATH = "lib/models/cnndatatokenized.pkl"


def fetch_html(url):
    try:
        return requests.get(url, timeout=10).text
    except Exception as e:
        print(f"Error fetching HTML: {e}")
        return ""

def extract_script_urls(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    scripts = []
    for tag in soup.find_all("script", src=True):
        src = tag["src"]
        full_url = urljoin(base_url, src)
        if not any(excl in full_url.lower() for excl in EXCLUDE_KEYWORDS):
            scripts.append(full_url)
    return scripts

def fetch_js(url):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            return res.text
    except Exception as e:
        print(f"Error fetching JS {url}: {e}")
    return ""

def extract_url_candidates(js_code):
    pattern = r"""(?:
        (?<!data:)
        (?:https?:\/\/)?
        (?:[\w\.-]+\.[a-z]{2,}|\/)
        [^\s"'<>\\\]\)]{2,}
    )"""
    return list(set(re.findall(pattern, js_code, re.VERBOSE | re.IGNORECASE)))

def extract_features(url):
    starts_with_slash = url.startswith("/")
    starts_with_http = url.startswith("http")

    num_slashes = 0
    contains_query = False
    contains_domain = False
    contains_extension = False

    length = len(url)

    dot_positions = []

    for i, ch in enumerate(url):
        if ch == "/":
            num_slashes += 1
        elif ch == "?":
            contains_query = True
        elif ch == ".":
            dot_positions.append(i)

    for pos in dot_positions:
        if pos + 2 < length and re.match(r"[a-z]{2,}", url[pos+1:pos+6]):
            contains_domain = True
            break

    for pos in dot_positions:
        end = url.find("?", pos)
        if end == -1:
            end = length
        ext_len = end - (pos + 1)
        if 2 <= ext_len <= 5 and url[pos+1:end].isalnum():
            contains_extension = True
            break

    return {
        "starts_with_slash": int(starts_with_slash),
        "starts_with_http": int(starts_with_http),
        "contains_domain": int(contains_domain),
        "num_slashes": num_slashes,
        "contains_query": int(contains_query),
        "contains_extension": int(contains_extension),
        "length": length,
    }

def check_model():
    if os.path.exists(CNN_MODEL_PATH) and os.path.exists(CNN_TOKEN_PATH):
        model = load_model(CNN_MODEL_PATH, compile=False)
        with open(CNN_TOKEN_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    else:
        print("No model data found.")
        return ""

def validate_urls(tokenizer, model, candidates, batch_size=1024, threshold=0.5):
    max_len = model.input_shape[1]
    valid_urls = []

    sequences = tokenizer.texts_to_sequences(candidates)

    for start in range(0, len(sequences), batch_size):
        batch_seq = sequences[start:start+batch_size]
        padded = pad_sequences(batch_seq, maxlen=max_len, padding='post')
        preds = model.predict(padded, verbose=0).flatten()
        mask = preds > threshold
        valid_urls.extend(np.array(candidates[start:start+batch_size])[mask])

    return valid_urls

def main():
    if len(sys.argv) < 2:
        print("Usage: python linkturn.py <URL>")
        return

    page_url = sys.argv[1]
    print(f"[+] Fetching page: {page_url}")
    html = fetch_html(page_url)

    script_urls = extract_script_urls(html, page_url)
    print(f"[+] Found {len(script_urls)} custom JS files")

    all_candidates = []
    for js_url in script_urls:
        print(f"\t-> {js_url}")
        js_code = fetch_js(js_url)
        candidates = extract_url_candidates(js_code)
        all_candidates.extend(candidates)

    print(f"[+] Discovered {len(all_candidates)} raw URL/URI candidates")
    print(f"[+] Deep analyzing with CNN algorithms, this is taking a few time...")
    model, tokenizer = check_model()
    valid_urls = validate_urls(tokenizer, model, all_candidates)
    print(f"[+] Extracted {len(valid_urls)} valid URLs/URIs")
    print(f"[+] Extracted URLs:")
    for url in valid_urls:
        if url.startswith("http"):
            print(f"\t- {url}")
    
    print(f"[+] Extracted URIs:")
    for url in valid_urls:
        if not url.startswith("http"):
            print(f"\t- {url}")

    print(f"[!] This result may contain false-positive coz the model should be trained with more dataset")
    print(f"[!] Use this tools for initial enumerations is recommended")

if __name__ == "__main__":
    main()

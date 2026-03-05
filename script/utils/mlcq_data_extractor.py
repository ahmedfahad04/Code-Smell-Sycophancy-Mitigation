import requests
import json
import time 
import os
from urllib.parse import quote
from tqdm import tqdm


TOKEN = os.getenv("GITHUB_TOKEN")
FALLBACK_BRANCHES = ["master", "main"]


def normalize_repo_name(repo_url: str) -> str:
    repo_url = repo_url.strip()
    if repo_url.startswith("git@github.com:"):
        return repo_url.split("git@github.com:")[-1].replace(".git", "")
    if repo_url.startswith("https://github.com/"):
        return repo_url.split("https://github.com/")[-1].replace(".git", "")
    if repo_url.startswith("http://github.com/"):
        return repo_url.split("http://github.com/")[-1].replace(".git", "")
    return repo_url.replace(".git", "")

def raw_url_from_link(link, headers):
    if not link:
        return None
    url = link.strip().split("#")[0]
    if not url:
        return None
    try:
        response = requests.head(url, headers=headers, allow_redirects=True)
        final_url = response.url
        if response.status_code >= 400:
            response = requests.get(url, headers=headers, allow_redirects=True)
            final_url = response.url
    except requests.RequestException:
        final_url = url
    if "github.com/" in final_url and "/blob/" in final_url:
        return final_url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")
    return None

def fetch_code_snippet(repo_url, commit_hash, file_path, start_line, end_line, request_count, link=None):

    if request_count > 4500:
        print("Reached 4500 requests, sleeping for 1 hour ...")
        time.sleep(3600)
        request_count = 0


    repo_name = normalize_repo_name(repo_url)
    
    raw_path = file_path.strip().lstrip('/')
    encoded_path = quote(raw_path, safe="/")
    raw_url = f"https://raw.githubusercontent.com/{repo_name}/{commit_hash}/{encoded_path}"

    headers = {'Authorization': f'token {TOKEN}'} if TOKEN else None
    
    response = requests.get(raw_url, headers=headers)
    
    if response.status_code == 200:
        lines = response.text.splitlines()
        
        code_snippet = "\n".join(lines[start_line-1:end_line])
        
        return code_snippet, request_count + 1
    if response.status_code == 404:
        for branch in FALLBACK_BRANCHES:
            fallback_url = f"https://raw.githubusercontent.com/{repo_name}/{branch}/{encoded_path}"
            fallback_response = requests.get(fallback_url, headers=headers)
            if fallback_response.status_code == 200:
                lines = fallback_response.text.splitlines()
                code_snippet = "\n".join(lines[start_line-1:end_line])
                print(f"Fallback to branch '{branch}' for {repo_name}:{raw_path}")
                return code_snippet, request_count + 1
        link_raw_url = raw_url_from_link(link, headers)
        if link_raw_url:
            link_response = requests.get(link_raw_url, headers=headers)
            if link_response.status_code == 200:
                lines = link_response.text.splitlines()
                code_snippet = "\n".join(lines[start_line-1:end_line])
                print(f"Fallback to link URL for {repo_name}:{raw_path}")
                return code_snippet, request_count + 1
    print(f"Failed to fetch code from {raw_url} (status code: {response.status_code})")
    return None, request_count + 1 
    
def process_csv_and_save_to_json(csv_file, json_file, batch_size=50):
    
    request_count = 0
    json_data = []
    counter = 0
    existing_ids = load_existing_ids(json_file)
    snippet_cache = {}
    failed_keys = set()
    failed_data = []
    failed_json_file = json_file.replace(".json", "_failed.json")

    with open(csv_file, 'r') as f:
        
        next(f)

        for _, line in tqdm(enumerate(f), desc="Fetching code snippets"):
            parts = [p.strip() for p in line.strip().split(";")]
            dataset_id, _, _, smell, severity, _, type, code_name, repo_url, commit_hash, file_path, start_line, end_line, link, _ = parts
            dataset_id = int(dataset_id)
            if dataset_id in existing_ids:
                continue
            
            start_line = int(start_line)
            end_line = int(end_line)
            
            cache_key = (repo_url, commit_hash, file_path, start_line, end_line)
            if cache_key in snippet_cache:
                code_snippet = snippet_cache[cache_key]
            elif cache_key in failed_keys:
                code_snippet = None
            else:
                code_snippet, request_count = fetch_code_snippet(
                    repo_url, commit_hash, file_path, start_line, end_line, request_count, link
                )
                if code_snippet:
                    snippet_cache[cache_key] = code_snippet
                else:
                    failed_keys.add(cache_key)
            
            if code_snippet:
                json_data.append({
                    "id": dataset_id,
                    "repo_url": repo_url,
                    "commit_hash": commit_hash,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "code_snippet": code_snippet,
                    "smell": smell,
                    "severity": severity
                })
            else:
                failed_data.append({
                    "id": dataset_id,
                    "repo_url": repo_url,
                    "commit_hash": commit_hash,
                    "file_path": file_path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "link": link,
                    "smell": smell,
                    "severity": severity
                })

            counter += 1 
            if counter % batch_size == 0:
                save_json_data(json_file, json_data)
                json_data = []
                if failed_data:
                    save_json_data(failed_json_file, failed_data)
                    failed_data = []

        if json_data:
            save_json_data(json_file, json_data)
        if failed_data:
            save_json_data(failed_json_file, failed_data)
        print(f"Completed processing. Data saved to {json_file}")
       
def save_json_data(json_file, json_data):

    try:
        with open(json_file, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []

    existing_data.extend(json_data)

    with open(json_file, 'w') as f:
        json.dump(existing_data, f, indent=4)

    print(f"Saved batch of {len(json_data)} entries to {json_file}")


def load_existing_ids(json_file):
    try:
        with open(json_file, 'r') as f:
            existing_data = json.load(f)
        return {entry.get("id") for entry in existing_data if "id" in entry}
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


if __name__ == '__main__':

    csv_file = "MLCQCodeSmellSamples.csv"
    json_file = "MLCQCodeSmellSamples_Updated.json"
    process_csv_and_save_to_json(csv_file, json_file)
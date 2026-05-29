"""
github_collector.py
-------------------
Collects GitHub repository data for the Econometrics & Macro Tooling
ecosystem. Uses the GitHub REST API to search repositories by topic/keyword
and enriches each result with additional signals.

Track B — Technology Innovation & Ecosystem Tracking
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}
BASE_URL = "https://api.github.com"

# ---------------------------------------------------------------------------
# Search queries — keywords and topics that define our ecosystem
# ---------------------------------------------------------------------------
SEARCH_QUERIES = [
    "topic:econometrics",
    "topic:macroeconomics",
    "topic:time-series",
    "topic:panel-data",
    "topic:dsge",
    "topic:cointegration",
    "topic:nowcasting",
    "topic:var-model",
    "topic:causal-inference language:python",
    "topic:structural-estimation",
    "econometrics in:name,description language:python",
    "macroeconomics in:name,description language:python",
    "dsge model in:name,description",
    "panel data econometrics in:name,description language:python",
    "time series econometrics in:name,description language:python",
]

# Repos with fewer than this many stars are excluded (noise filter)
MIN_STARS = 5


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, retries: int = 3) -> dict | list | None:
    """GET request with rate-limit handling and retries."""
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, params=params, timeout=15)

            if response.status_code == 403:
                reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset_time - int(time.time()), 1)
                print(f"  Rate limit hit. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code == 404:
                return None

            if response.status_code == 202:
                # GitHub is computing stats — wait and retry
                time.sleep(3)
                continue

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"  Request error (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_repositories(query: str, max_results: int = 100) -> list[dict]:
    """Search GitHub repositories and return raw items."""
    results = []
    per_page = 30
    page = 1

    while len(results) < max_results:
        data = _get(
            f"{BASE_URL}/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
        )

        if not data or "items" not in data:
            break

        items = data["items"]
        if not items:
            break

        results.extend(items)
        page += 1

        # Respect secondary rate limits
        time.sleep(1)

        if len(results) >= data.get("total_count", 0):
            break

    return results[:max_results]


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def get_contributors_count(owner: str, repo: str) -> int:
    """Return total number of contributors (capped at 500 for performance)."""
    data = _get(
        f"{BASE_URL}/repos/{owner}/{repo}/contributors",
        params={"per_page": 1, "anon": "true"},
    )
    if data is None:
        return 0
    # GitHub returns paginated results — check Link header via a head request
    resp = requests.get(
        f"{BASE_URL}/repos/{owner}/{repo}/contributors",
        headers=HEADERS,
        params={"per_page": 1, "anon": "true"},
        timeout=15,
    )
    link = resp.headers.get("Link", "")
    if 'rel="last"' in link:
        import re
        match = re.search(r'page=(\d+)>; rel="last"', link)
        if match:
            return int(match.group(1))
    return len(data) if isinstance(data, list) else 0


def get_weekly_commit_avg(owner: str, repo: str) -> float:
    """Return average weekly commits over the last 52 weeks."""
    data = _get(f"{BASE_URL}/repos/{owner}/{repo}/stats/commit_activity")
    if not data or not isinstance(data, list):
        return 0.0
    totals = [week.get("total", 0) for week in data]
    return round(sum(totals) / len(totals), 2) if totals else 0.0


def get_releases_count(owner: str, repo: str) -> int:
    """Return total number of releases."""
    data = _get(
        f"{BASE_URL}/repos/{owner}/{repo}/releases",
        params={"per_page": 1},
    )
    if data is None:
        return 0
    resp = requests.get(
        f"{BASE_URL}/repos/{owner}/{repo}/releases",
        headers=HEADERS,
        params={"per_page": 1},
        timeout=15,
    )
    link = resp.headers.get("Link", "")
    if 'rel="last"' in link:
        import re
        match = re.search(r'page=(\d+)>; rel="last"', link)
        if match:
            return int(match.group(1))
    return len(data) if isinstance(data, list) else 0


def get_closed_issues_count(owner: str, repo: str) -> int:
    """Return number of closed issues (proxy for community health)."""
    data = _get(
        f"{BASE_URL}/repos/{owner}/{repo}/issues",
        params={"state": "closed", "per_page": 1},
    )
    if data is None:
        return 0
    resp = requests.get(
        f"{BASE_URL}/repos/{owner}/{repo}/issues",
        headers=HEADERS,
        params={"state": "closed", "per_page": 1},
        timeout=15,
    )
    link = resp.headers.get("Link", "")
    if 'rel="last"' in link:
        import re
        match = re.search(r'page=(\d+)>; rel="last"', link)
        if match:
            return int(match.group(1))
    return len(data) if isinstance(data, list) else 0


def has_ci_workflow(owner: str, repo: str) -> bool:
    """Check whether the repository contains GitHub Actions workflows."""
    data = _get(f"{BASE_URL}/repos/{owner}/{repo}/contents/.github/workflows")
    return isinstance(data, list) and len(data) > 0


def get_readme_length(owner: str, repo: str) -> int:
    """Return README character length (0 if not found)."""
    import base64
    data = _get(f"{BASE_URL}/repos/{owner}/{repo}/readme")
    if not data or "content" not in data:
        return 0
    try:
        content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
        return len(content)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Full repo enrichment
# ---------------------------------------------------------------------------

def enrich_repository(item: dict) -> dict:
    """Extract all signals from a raw GitHub search item."""
    owner = item["owner"]["login"]
    repo = item["name"]

    now = datetime.now(timezone.utc)
    created_at = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
    pushed_at = datetime.fromisoformat(item["pushed_at"].replace("Z", "+00:00"))

    repo_age_days = (now - created_at).days
    days_since_last_push = (now - pushed_at).days

    return {
        # Identity
        "full_name": item["full_name"],
        "owner": owner,
        "repo": repo,
        "description": item.get("description", ""),
        "url": item["html_url"],
        "language": item.get("language", ""),
        "topics": "|".join(item.get("topics", [])),
        "license": item.get("license", {}).get("spdx_id", "") if item.get("license") else "",

        # --- Signal 1: Stars (popularity proxy) ---
        "stars": item["stargazers_count"],

        # --- Signal 2: Forks (adoption proxy) ---
        "forks": item["forks_count"],

        # --- Signal 3: Open issues (community activity) ---
        "open_issues": item["open_issues_count"],

        # --- Signal 4: Repository age in days (maturity proxy) ---
        "repo_age_days": repo_age_days,

        # --- Signal 5: Days since last push (activity recency) ---
        "days_since_last_push": days_since_last_push,

        # --- Signal 6: Watchers (sustained interest) ---
        "watchers": item["watchers_count"],

        # --- Signal 7: Contributors (ecosystem breadth) ---
        "contributors_count": get_contributors_count(owner, repo),

        # --- Signal 8: Weekly commit average last 52 weeks (development velocity) ---
        "weekly_commit_avg": get_weekly_commit_avg(owner, repo),

        # --- Signal 9: Releases count (production-readiness) ---
        "releases_count": get_releases_count(owner, repo),

        # --- Signal 10: Closed issues (responsiveness / health) ---
        "closed_issues_count": get_closed_issues_count(owner, repo),

        # --- Signal 11: CI/CD presence (engineering maturity) ---
        "has_ci": has_ci_workflow(owner, repo),

        # --- Signal 12: README length (documentation quality) ---
        "readme_length": get_readme_length(owner, repo),

        # Metadata
        "created_at": item["created_at"],
        "pushed_at": item["pushed_at"],
        "collected_at": now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect(max_per_query: int = 100, output_path: str = "data/raw/repositories.csv"):
    """
    Run full collection pipeline:
    1. Search repos for all queries
    2. Deduplicate
    3. Filter by MIN_STARS
    4. Enrich with additional signals
    5. Save to CSV
    """
    print("=" * 60)
    print("GitHub Econometrics & Macro Tooling — Data Collection")
    print("=" * 60)

    # Step 1: Search
    raw_items = {}
    for query in SEARCH_QUERIES:
        print(f"\nSearching: '{query}'")
        items = search_repositories(query, max_results=max_per_query)
        print(f"  Found {len(items)} results")
        for item in items:
            raw_items[item["full_name"]] = item  # deduplicate by full_name

    print(f"\nTotal unique repos before filtering: {len(raw_items)}")

    # Step 2: Filter by minimum stars
    filtered = {k: v for k, v in raw_items.items() if v["stargazers_count"] >= MIN_STARS}
    print(f"Total after star filter (>= {MIN_STARS}): {len(filtered)}")

    # Step 3: Enrich
    print("\nEnriching repositories with additional signals...")
    records = []
    for full_name, item in tqdm(filtered.items()):
        try:
            record = enrich_repository(item)
            records.append(record)
            time.sleep(0.5)  # be a good API citizen
        except Exception as e:
            print(f"  Error enriching {full_name}: {e}")
            continue

    # Step 4: Save
    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nCollection complete.")
    print(f"  Repositories collected: {len(df)}")
    print(f"  Saved to: {output_path}")
    print(f"  Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    df = collect()
    print(df.head())

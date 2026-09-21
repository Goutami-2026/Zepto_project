# %% [markdown]
# # Data Pipeline: books.toscrape.com -> clean -> SQLite
#
# Scrapes books across >=3 categories, cleans fields into proper types,
# converts GBP -> INR at a fixed baseline rate, and loads into a
# normalized SQLite database (categories <-> books, PK/FK).

# %%
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = BASE_URL + "catalogue/"
GBP_TO_INR = 105.50  # fixed, project-defined baseline conversion rate (no live lookup)

DB_PATH = Path(__file__).parent / "books.db"

RATING_WORDS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

# %% [markdown]
# ## Step 1 — discover category pages
#
# The site's left sidebar lists every category. We pull that list once from
# the home page, then visit each category's own paginated listing.

# %%
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_category_links(n_categories: int = 4) -> list[dict]:
    """Return the first n_categories (name, url) pairs from the sidebar nav."""
    soup = get_soup(BASE_URL)
    nav_links = soup.select("div.side_categories ul li ul li a")
    categories = []
    for a in nav_links[:n_categories]:
        name = a.get_text(strip=True)
        url = BASE_URL + a["href"]
        categories.append({"name": name, "url": url})
    return categories


# %% [markdown]
# ## Step 2 — scrape all books within a category (handles pagination)

# %%
def scrape_category(category_name: str, category_url: str) -> list[dict]:
    rows = []
    url = category_url
    while url:
        soup = get_soup(url)
        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"]
            price_text = article.select_one("p.price_color").get_text(strip=True)
            rating_class = article.select_one("p.star-rating")["class"]
            rating_word = [c for c in rating_class if c != "star-rating"][0]
            availability = article.select_one("p.instock.availability").get_text(strip=True)
            rows.append(
                {
                    "title": title,
                    "price_text": price_text,
                    "rating_word": rating_word,
                    "availability_text": availability,
                    "category": category_name,
                }
            )
        next_link = soup.select_one("li.next a")
        url = (url.rsplit("/", 1)[0] + "/" + next_link["href"]) if next_link else None
        time.sleep(0.2)  # be polite to the scraping-practice site
    return rows


# %%
categories = get_category_links(n_categories=4)
print("Scraping categories:", [c["name"] for c in categories])

all_rows = []
for cat in categories:
    cat_rows = scrape_category(cat["name"], cat["url"])
    print(f"  {cat['name']}: {len(cat_rows)} books")
    all_rows.extend(cat_rows)

raw_df = pd.DataFrame(all_rows)
print(f"\nTotal scraped rows: {len(raw_df)}")
assert len(raw_df) >= 60, "Need at least 60 books total per assignment spec"
raw_df.head()

# %% [markdown]
# ## Step 3 — clean fields into proper types
#
# - `price_gbp`: strip currency symbol -> float
# - `rating`: word ("Three") -> int (1-5)
# - `in_stock`: availability text -> bool
# - Any row that fails to parse a required field is dropped (see justification
#   below) rather than crashing the pipeline.

# %%
def parse_price(text: str) -> float | None:
    match = re.search(r"[\d.]+", text)
    return float(match.group()) if match else None


def parse_in_stock(text: str) -> bool | None:
    if "In stock" in text:
        return True
    if "Out of stock" in text.lower() or "unavailable" in text.lower():
        return False
    return None


clean_df = raw_df.copy()
clean_df["price_gbp"] = clean_df["price_text"].apply(parse_price)
clean_df["rating"] = clean_df["rating_word"].map(RATING_WORDS)
clean_df["in_stock"] = clean_df["availability_text"].apply(parse_in_stock)

before = len(clean_df)
# Design decision: books.toscrape.com is a clean, purpose-built practice site,
# so unparseable rows (if any) are rare/anomalous rather than a systematic
# gap. We DROP such rows rather than median-imputing price/rating, since
# imputing a book's price or star rating would fabricate catalog facts that
# don't reflect the real listing. (If a required numeric column had a high
# missing rate, median imputation would be the better call -- not the case
# here.)
clean_df = clean_df.dropna(subset=["price_gbp", "rating", "in_stock"])
dropped = before - len(clean_df)
print(f"Dropped {dropped} unparseable rows out of {before}")

clean_df["in_stock"] = clean_df["in_stock"].astype(bool)
clean_df["rating"] = clean_df["rating"].astype(int)

# %% [markdown]
# ## Step 4 — currency conversion (fixed baseline rate)
#
# 1 GBP = 105.50 INR — a fixed, project-defined constant. No API call, no
# date reference; see README for why.

# %%
clean_df["price_inr"] = (clean_df["price_gbp"] * GBP_TO_INR).round(2)
clean_df = clean_df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]
clean_df.head()

# %% [markdown]
# ## Step 5 — normalized SQLite schema (categories <-> books, PK/FK)

# %%
def build_database(df: pd.DataFrame, db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()  # rebuild from scratch every run
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        )
        """
    )

    category_names = sorted(df["category"].unique())
    cur.executemany(
        "INSERT INTO categories (category_name) VALUES (?)",
        [(name,) for name in category_names],
    )
    conn.commit()

    cat_id_map = dict(
        cur.execute("SELECT category_name, category_id FROM categories").fetchall()
    )

    book_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            row.rating,
            int(row.in_stock),
            cat_id_map[row.category],
        )
        for row in df.itertuples(index=False)
    ]
    cur.executemany(
        """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        book_rows,
    )
    conn.commit()
    conn.close()


build_database(clean_df, DB_PATH)
print(f"Database built at {DB_PATH} with {len(clean_df)} books across {clean_df['category'].nunique()} categories")

# %% [markdown]
# Next: run `queries.py` to execute the required SQL queries and their
# pandas equivalents against this database.

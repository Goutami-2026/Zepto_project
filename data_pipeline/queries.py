# %% [markdown]
# # SQL + pandas queries against books.db
#
# Runs >=5 SQL queries covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT,
# IN/BETWEEN, and a JOIN — then reproduces two of them with pandas
# (`pd.read_sql` and, for the join, `pd.merge` on in-memory DataFrames)
# to show both approaches agree.
#
# Run scrape_and_load.py first so data_pipeline/books.db exists.

# %%
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "books.db"
conn = sqlite3.connect(DB_PATH)


def run(label: str, sql: str) -> pd.DataFrame:
    print(f"\n--- {label} ---")
    print(sql.strip())
    df = pd.read_sql(sql, conn)
    print(df.to_string(index=False))
    return df


# %% [markdown]
# ## Query 1 — SELECT / WHERE
# Books priced under INR 1000.

# %%
q1_sql = """
SELECT title, price_inr, rating
FROM books
WHERE price_inr < 1000
"""
q1_df = run("Q1: SELECT/WHERE - books under INR 1000", q1_sql)

# %% [markdown]
# ## Query 2 — ORDER BY + LIMIT
# 10 most expensive books (by price_inr).

# %%
q2_sql = """
SELECT title, price_inr
FROM books
ORDER BY price_inr DESC
LIMIT 10
"""
q2_df = run("Q2: ORDER BY/LIMIT - 10 most expensive books", q2_sql)

# %% [markdown]
# ## Query 3 — DISTINCT
# Distinct rating values present in the dataset.

# %%
q3_sql = """
SELECT DISTINCT rating
FROM books
ORDER BY rating
"""
q3_df = run("Q3: DISTINCT - rating values in use", q3_sql)

# %% [markdown]
# ## Query 4 — IN / BETWEEN
# Books rated 4 or 5 stars, priced between INR 500 and INR 2000.

# %%
q4_sql = """
SELECT title, rating, price_inr
FROM books
WHERE rating IN (4, 5)
  AND price_inr BETWEEN 500 AND 2000
ORDER BY price_inr
"""
q4_df = run("Q4: IN/BETWEEN - 4-5 star books priced 500-2000 INR", q4_sql)

# %% [markdown]
# ## Query 5 — JOIN
# Top 10 highest-rated books per category (JOIN books <-> categories).

# %%
q5_sql = """
SELECT c.category_name, b.title, b.rating, b.price_inr
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY b.rating DESC, b.price_inr DESC
LIMIT 10
"""
q5_df = run("Q5: JOIN - top 10 highest-rated books (with category)", q5_sql)

# %% [markdown]
# ## Query 6 (bonus) — in-stock count per category
# A small aggregate query beyond the required 5, useful for the README.

# %%
q6_sql = """
SELECT c.category_name, COUNT(*) AS in_stock_count
FROM books b
JOIN categories c ON b.category_id = c.category_id
WHERE b.in_stock = 1
GROUP BY c.category_name
"""
q6_df = run("Q6 (bonus): in-stock book count per category", q6_sql)

# %% [markdown]
# ## pandas equivalents
#
# Read two query results back via `pd.read_sql` (already done above for
# every query, since `run()` uses it), and separately reproduce the JOIN
# query (Q5) using `pd.merge` on in-memory DataFrames pulled with plain
# `SELECT *` — no SQL JOIN involved this time — to prove both approaches
# agree.

# %%
books_df = pd.read_sql("SELECT * FROM books", conn)
categories_df = pd.read_sql("SELECT * FROM categories", conn)

merged_df = books_df.merge(categories_df, on="category_id", how="inner")
q5_via_pandas = (
    merged_df.sort_values(["rating", "price_inr"], ascending=[False, False])
    .loc[:, ["category_name", "title", "rating", "price_inr"]]
    .head(10)
    .reset_index(drop=True)
)

print("\n--- Q5 reproduced via pd.merge (no SQL JOIN) ---")
print(q5_via_pandas.to_string(index=False))

# %%
sql_result = q5_df.reset_index(drop=True)
pandas_result = q5_via_pandas.reset_index(drop=True)
are_equal = sql_result.equals(pandas_result)
print(f"\nSQL JOIN result matches pd.merge result: {are_equal}")
assert are_equal, "SQL and pandas results should match exactly"

# %%
conn.close()
print("\nAll queries executed successfully; SQL and pandas outputs verified equal.")
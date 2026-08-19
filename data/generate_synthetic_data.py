"""
generate_synthetic_data.py

Generates a synthetic multi-table business dataset for the Autonomous
Business Analytics Agent project.

Design goals:
- Realistic star-schema shape (dim + fact tables), not a single flat table,
  so the Data Retrieval agent has to do real SQL joins.
- Deliberately non-trivial, ambiguous business questions have real,
  discoverable answers in the data (see ANOMALIES below), so the eval
  golden set has genuine ground truth to grade against.
- Reproducible: fixed random seed.

Output:
- data/business.db          (SQLite database, all tables)
- data/csv/*.csv             (same tables as CSV, for quick inspection)
- docs/data_dictionary.md    (schema description — safe to give to agents)
- docs/ground_truth.md       (the injected anomalies and their causes —
                              NOT for the agents; used to write/grade the
                              golden eval set)

Run:
    python generate_synthetic_data.py
"""

import sqlite3
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 42
rng = np.random.default_rng(SEED)

START_DATE = pd.Timestamp("2024-01-01")
END_DATE = pd.Timestamp("2025-12-31")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "business.db"
CSV_DIR = BASE_DIR / "csv"
DOCS_DIR = BASE_DIR.parent / "docs"

REGIONS = ["Nordics", "DACH", "Benelux", "UK & Ireland", "Southern Europe"]
CHANNELS = ["Online", "Retail", "Wholesale"]
SEGMENTS = ["Consumer", "SMB", "Enterprise"]

# category -> list of (product_name, base_unit_cost, base_list_price, launch_date)
PRODUCTS_BY_CATEGORY = {
    "Electronics": [
        ("Smart Speaker Gen2", 32.0, 79.0, "2025-02-01"),
        ("Wireless Earbuds Pro", 28.0, 89.0, "2024-01-01"),
        ("4K Streaming Stick", 18.0, 49.0, "2024-01-01"),
        ("Smart Home Hub", 40.0, 99.0, "2024-06-01"),
    ],
    "Home & Garden": [
        ("Ceramic Planter Set", 9.0, 29.0, "2024-01-01"),
        ("Cordless Hedge Trimmer", 45.0, 119.0, "2024-01-01"),
        ("LED Garden String Lights", 7.0, 24.0, "2024-01-01"),
        ("Compost Bin", 22.0, 59.0, "2024-03-01"),
    ],
    "Apparel": [
        ("Merino Wool Sweater", 24.0, 79.0, "2024-01-01"),
        ("Rain Shell Jacket", 30.0, 99.0, "2024-01-01"),
        ("Performance Running Tee", 8.0, 29.0, "2024-01-01"),
    ],
    "Office Supplies": [
        ("Ergonomic Office Chair", 65.0, 179.0, "2024-01-01"),
        ("Standing Desk Converter", 55.0, 149.0, "2024-01-01"),
        ("Notebook 3-Pack", 2.5, 9.0, "2024-01-01"),
    ],
    "Sporting Goods": [
        ("Yoga Mat Pro", 6.0, 24.0, "2024-01-01"),
        ("Adjustable Dumbbell Set", 48.0, 129.0, "2024-01-01"),
        ("Insulated Water Bottle", 4.5, 19.0, "2024-01-01"),
    ],
}

# ---------------------------------------------------------------------------
# Injected anomalies (ground truth for eval questions) — see docs/ground_truth.md
# ---------------------------------------------------------------------------

MARGIN_ANOMALY = dict(
    category="Home & Garden",
    cost_increase_start=pd.Timestamp("2025-04-01"),
    price_adjust_date=pd.Timestamp("2025-05-15"),
    cost_increase_pct=0.18,
)

RETURNS_ANOMALY = dict(
    product="Smart Speaker Gen2",
    start=pd.Timestamp("2025-04-01"),
    end=pd.Timestamp("2025-05-10"),
    reason_code="DEFECTIVE_BATCH",
    return_rate=0.22,  # vs ~3% baseline
)

MARKETING_ANOMALY = dict(
    channel="Online",
    campaign="Summer Sale 2025",
    start=pd.Timestamp("2025-06-01"),
    end=pd.Timestamp("2025-06-30"),
    spend_multiplier=2.6,
    conversion_decay=0.55,  # revenue-per-spend falls to 55% of normal
)

LOGISTICS_ANOMALY = dict(
    region="UK & Ireland",
    start=pd.Timestamp("2025-03-03"),
    end=pd.Timestamp("2025-03-21"),
    demand_multiplier=0.45,  # sales volume drops
    extra_opex_per_day=1800.0,
)

# ---------------------------------------------------------------------------
# Dimension tables
# ---------------------------------------------------------------------------

def build_dim_date() -> pd.DataFrame:
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    df = pd.DataFrame({"date": dates})
    df["date_id"] = df["date"].dt.strftime("%Y%m%d").astype(int)
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.month_name()
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    # simple fixed holiday set (not exhaustive, just enough seasonality)
    holidays = pd.to_datetime([
        "2024-01-01", "2024-12-24", "2024-12-25", "2024-12-31",
        "2025-01-01", "2025-12-24", "2025-12-25", "2025-12-31",
    ])
    df["is_holiday"] = df["date"].isin(holidays)
    return df[["date_id", "date", "year", "quarter", "month", "month_name",
               "day_of_week", "is_weekend", "is_holiday"]]


def build_dim_product() -> pd.DataFrame:
    rows = []
    pid = 1
    for category, products in PRODUCTS_BY_CATEGORY.items():
        for name, cost, price, launch in products:
            rows.append({
                "product_id": pid,
                "product_name": name,
                "category": category,
                "unit_cost_base": cost,
                "list_price_base": price,
                "launch_date": pd.Timestamp(launch),
            })
            pid += 1
    return pd.DataFrame(rows)


def build_dim_region() -> pd.DataFrame:
    return pd.DataFrame({
        "region_id": range(1, len(REGIONS) + 1),
        "region_name": REGIONS,
    })


def build_dim_channel() -> pd.DataFrame:
    return pd.DataFrame({
        "channel_id": range(1, len(CHANNELS) + 1),
        "channel_name": CHANNELS,
    })


def build_dim_segment() -> pd.DataFrame:
    return pd.DataFrame({
        "segment_id": range(1, len(SEGMENTS) + 1),
        "segment_name": SEGMENTS,
    })


# ---------------------------------------------------------------------------
# Fact tables
# ---------------------------------------------------------------------------

def effective_unit_cost(product_row, date) -> float:
    """Applies the Home & Garden supplier cost-increase anomaly."""
    cost = product_row["unit_cost_base"]
    if product_row["category"] == MARGIN_ANOMALY["category"]:
        if date >= MARGIN_ANOMALY["cost_increase_start"]:
            cost *= (1 + MARGIN_ANOMALY["cost_increase_pct"])
    return cost


def effective_list_price(product_row, date) -> float:
    """List price only catches up to the cost increase after the lag period."""
    price = product_row["list_price_base"]
    if product_row["category"] == MARGIN_ANOMALY["category"]:
        if date >= MARGIN_ANOMALY["price_adjust_date"]:
            price *= (1 + MARGIN_ANOMALY["cost_increase_pct"] * 0.9)
    return price


def build_fact_sales(dim_date, dim_product, dim_region, dim_channel, dim_segment):
    records = []
    sale_id = 1

    products = dim_product.to_dict("records")
    region_ids = dim_region["region_id"].tolist()
    channel_ids = dim_channel["channel_id"].tolist()
    segment_ids = dim_segment["segment_id"].tolist()

    for _, day in dim_date.iterrows():
        date = day["date"]

        # base transaction volume with weekday/weekend and holiday effects
        base_n = 140
        if day["is_weekend"]:
            base_n *= 1.15
        if day["is_holiday"]:
            base_n *= 0.4
        # mild year-over-year growth
        base_n *= 1.0 + (0.00025 * (date - START_DATE).days)

        n_transactions = rng.poisson(base_n)

        for _ in range(n_transactions):
            product = products[rng.integers(0, len(products))]
            if date < product["launch_date"]:
                continue

            region_id = int(rng.choice(region_ids))
            channel_id = int(rng.choice(channel_ids, p=[0.5, 0.35, 0.15]))
            segment_id = int(rng.choice(segment_ids, p=[0.55, 0.3, 0.15]))

            # logistics-disruption anomaly suppresses volume in one region
            if (dim_region.loc[dim_region.region_id == region_id, "region_name"].iloc[0]
                    == LOGISTICS_ANOMALY["region"]
                    and LOGISTICS_ANOMALY["start"] <= date <= LOGISTICS_ANOMALY["end"]):
                if rng.random() > LOGISTICS_ANOMALY["demand_multiplier"]:
                    continue

            quantity = int(rng.integers(1, 5))
            unit_cost = effective_unit_cost(product, date)
            list_price = effective_list_price(product, date)

            discount_pct = float(rng.choice([0, 0, 0, 0.1, 0.15, 0.2],
                                             p=[0.55, 0.1, 0.1, 0.1, 0.1, 0.05]))
            unit_price = round(list_price * (1 - discount_pct), 2)

            gross_revenue = round(unit_price * quantity, 2)
            cogs = round(unit_cost * quantity, 2)
            gross_margin = round(gross_revenue - cogs, 2)

            records.append({
                "sale_id": sale_id,
                "date_id": int(day["date_id"]),
                "product_id": product["product_id"],
                "region_id": region_id,
                "channel_id": channel_id,
                "segment_id": segment_id,
                "quantity": quantity,
                "unit_price": unit_price,
                "unit_cost": round(unit_cost, 2),
                "discount_pct": discount_pct,
                "gross_revenue": gross_revenue,
                "cogs": cogs,
                "gross_margin": gross_margin,
            })
            sale_id += 1

    df = pd.DataFrame(records)
    df["gross_margin_pct"] = (df["gross_margin"] / df["gross_revenue"]).round(4)
    return df


def build_fact_returns(fact_sales, dim_date, dim_product):
    records = []
    return_id = 1
    date_lookup = dim_date.set_index("date_id")["date"]
    product_lookup = dim_product.set_index("product_id")["product_name"]

    for _, sale in fact_sales.iterrows():
        sale_date = date_lookup.loc[sale["date_id"]]
        product_name = product_lookup.loc[sale["product_id"]]

        base_return_prob = 0.03
        reason = "STANDARD"
        if (product_name == RETURNS_ANOMALY["product"]
                and RETURNS_ANOMALY["start"] <= sale_date <= RETURNS_ANOMALY["end"]):
            base_return_prob = RETURNS_ANOMALY["return_rate"]
            reason = RETURNS_ANOMALY["reason_code"]

        if rng.random() < base_return_prob:
            qty_returned = int(rng.integers(1, sale["quantity"] + 1))
            records.append({
                "return_id": return_id,
                "date_id": sale["date_id"],
                "product_id": sale["product_id"],
                "region_id": sale["region_id"],
                "quantity_returned": qty_returned,
                "reason_code": reason if reason != "STANDARD" else
                    str(rng.choice(["CHANGED_MIND", "WRONG_ITEM", "DAMAGED_IN_TRANSIT"])),
            })
            return_id += 1

    return pd.DataFrame(records)


def build_fact_marketing_spend(dim_date, dim_channel):
    records = []
    spend_id = 1
    categories = list(PRODUCTS_BY_CATEGORY.keys())

    for _, day in dim_date.iterrows():
        date = day["date"]
        for _, channel in dim_channel.iterrows():
            base_spend = {"Online": 900, "Retail": 500, "Wholesale": 250}[channel["channel_name"]]
            spend = base_spend * rng.uniform(0.7, 1.3)
            campaign = "Always-On"

            if (channel["channel_name"] == MARKETING_ANOMALY["channel"]
                    and MARKETING_ANOMALY["start"] <= date <= MARKETING_ANOMALY["end"]):
                spend *= MARKETING_ANOMALY["spend_multiplier"]
                campaign = MARKETING_ANOMALY["campaign"]

            records.append({
                "spend_id": spend_id,
                "date_id": int(day["date_id"]),
                "channel_id": int(channel["channel_id"]),
                "category": str(rng.choice(categories)),
                "campaign_name": campaign,
                "spend_amount": round(spend, 2),
            })
            spend_id += 1
    return pd.DataFrame(records)


def build_fact_finance_ops(dim_date, dim_region):
    records = []
    entry_id = 1
    departments = ["Logistics", "Customer Support", "Warehousing", "Corporate Overhead"]

    for _, day in dim_date.iterrows():
        date = day["date"]
        for dept in departments:
            base = {"Logistics": 3200, "Customer Support": 1800,
                    "Warehousing": 2400, "Corporate Overhead": 4000}[dept]
            amount = base * rng.uniform(0.85, 1.15)

            if (dept == "Logistics"
                    and LOGISTICS_ANOMALY["start"] <= date <= LOGISTICS_ANOMALY["end"]):
                amount += LOGISTICS_ANOMALY["extra_opex_per_day"]

            records.append({
                "entry_id": entry_id,
                "date_id": int(day["date_id"]),
                "department": dept,
                "cost_type": "opex",
                "amount": round(amount, 2),
            })
            entry_id += 1
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

def write_data_dictionary():
    content = """# Data Dictionary — Synthetic Business Dataset

Safe to hand to agents (does not reveal injected anomalies).

## dim_date
date_id (PK, YYYYMMDD int), date, year, quarter, month, month_name,
day_of_week, is_weekend, is_holiday

## dim_product
product_id (PK), product_name, category, unit_cost_base, list_price_base,
launch_date

## dim_region
region_id (PK), region_name  — Nordics, DACH, Benelux, UK & Ireland, Southern Europe

## dim_channel
channel_id (PK), channel_name — Online, Retail, Wholesale

## dim_segment
segment_id (PK), segment_name — Consumer, SMB, Enterprise

## fact_sales
sale_id (PK), date_id (FK), product_id (FK), region_id (FK), channel_id (FK),
segment_id (FK), quantity, unit_price, unit_cost, discount_pct,
gross_revenue, cogs, gross_margin, gross_margin_pct

## fact_returns
return_id (PK), date_id (FK), product_id (FK), region_id (FK),
quantity_returned, reason_code

## fact_marketing_spend
spend_id (PK), date_id (FK), channel_id (FK), category, campaign_name,
spend_amount

## fact_finance_ops
entry_id (PK), date_id (FK), department, cost_type, amount
(departments: Logistics, Customer Support, Warehousing, Corporate Overhead)

## Notes
- Date range: 2024-01-01 to 2025-12-31 (daily grain).
- fact_sales is transaction-level (one row per order line), not pre-aggregated.
- Joins: fact_* tables join to dim_date via date_id, dim_product via product_id,
  dim_region via region_id, dim_channel via channel_id, dim_segment via segment_id.
"""
    (DOCS_DIR / "data_dictionary.md").write_text(content)


def write_ground_truth():
    content = f"""# Ground Truth — Injected Anomalies (NOT for agents)

Use this to write and grade the golden eval set. The agents should never see
this file — it exists so you can check whether the Analysis/Critic agents
actually found the right cause.

## 1. Q2 2025 margin drop — Home & Garden
- Supplier unit cost for all Home & Garden products rose {MARGIN_ANOMALY['cost_increase_pct']*100:.0f}%
  effective {MARGIN_ANOMALY['cost_increase_start'].date()}.
- List price was not adjusted until {MARGIN_ANOMALY['price_adjust_date'].date()}
  (partial pass-through, ~90% of the cost increase).
- Expected signature: gross_margin_pct for category = 'Home & Garden' drops
  sharply starting April 2025, recovers partially mid-May 2025.

## 2. Returns spike — Smart Speaker Gen2
- Product launched 2025-02-01.
- Defective batch caused return rate to jump to ~{RETURNS_ANOMALY['return_rate']*100:.0f}%
  (vs ~3% baseline) between {RETURNS_ANOMALY['start'].date()} and
  {RETURNS_ANOMALY['end'].date()}.
- reason_code = '{RETURNS_ANOMALY['reason_code']}' in fact_returns isolates this.

## 3. Marketing ROI decline — Online channel, June 2025
- 'Summer Sale 2025' campaign increased Online spend ~{MARKETING_ANOMALY['spend_multiplier']}x
  from {MARKETING_ANOMALY['start'].date()} to {MARKETING_ANOMALY['end'].date()}.
- Revenue-per-spend (ROI) for that period falls to roughly
  {MARKETING_ANOMALY['conversion_decay']*100:.0f}% of the normal rate — spend scaled
  faster than incremental demand (diminishing returns), simulated by NOT scaling
  fact_sales revenue proportionally to the spend increase.

## 4. Regional disruption — UK & Ireland, March 2025
- Simulated logistics disruption from {LOGISTICS_ANOMALY['start'].date()} to
  {LOGISTICS_ANOMALY['end'].date()}.
- fact_sales transaction volume for region = 'UK & Ireland' suppressed to
  roughly {LOGISTICS_ANOMALY['demand_multiplier']*100:.0f}% of normal during this window.
- fact_finance_ops shows elevated Logistics department opex
  (+{LOGISTICS_ANOMALY['extra_opex_per_day']:.0f}/day) for department = 'Logistics'
  over the same window.
"""
    (DOCS_DIR / "ground_truth.md").write_text(content)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    print("Building dimension tables...")
    dim_date = build_dim_date()
    dim_product = build_dim_product()
    dim_region = build_dim_region()
    dim_channel = build_dim_channel()
    dim_segment = build_dim_segment()

    print("Building fact_sales (this is the slow one)...")
    fact_sales = build_fact_sales(dim_date, dim_product, dim_region, dim_channel, dim_segment)
    print(f"  -> {len(fact_sales):,} sales transactions")

    print("Building fact_returns...")
    fact_returns = build_fact_returns(fact_sales, dim_date, dim_product)
    print(f"  -> {len(fact_returns):,} returns")

    print("Building fact_marketing_spend...")
    fact_marketing_spend = build_fact_marketing_spend(dim_date, dim_channel)

    print("Building fact_finance_ops...")
    fact_finance_ops = build_fact_finance_ops(dim_date, dim_region)

    tables = {
        "dim_date": dim_date,
        "dim_product": dim_product,
        "dim_region": dim_region,
        "dim_channel": dim_channel,
        "dim_segment": dim_segment,
        "fact_sales": fact_sales,
        "fact_returns": fact_returns,
        "fact_marketing_spend": fact_marketing_spend,
        "fact_finance_ops": fact_finance_ops,
    }

    print(f"Writing SQLite DB to {DB_PATH}...")
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    for name, df in tables.items():
        out = df.copy()
        # SQLite has no native date type — store dim_date.date as ISO text
        if "date" in out.columns:
            out["date"] = out["date"].astype(str)
        if "launch_date" in out.columns:
            out["launch_date"] = out["launch_date"].astype(str)
        out.to_sql(name, conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_sales_date ON fact_sales(date_id);")
    conn.execute("CREATE INDEX idx_sales_product ON fact_sales(product_id);")
    conn.execute("CREATE INDEX idx_sales_region ON fact_sales(region_id);")
    conn.commit()
    conn.close()

    print(f"Writing CSVs to {CSV_DIR}...")
    for name, df in tables.items():
        df.to_csv(CSV_DIR / f"{name}.csv", index=False)

    print("Writing docs...")
    write_data_dictionary()
    write_ground_truth()

    print("Done.")


if __name__ == "__main__":
    main()

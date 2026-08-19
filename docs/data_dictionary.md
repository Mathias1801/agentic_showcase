# Data Dictionary — Synthetic Business Dataset

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

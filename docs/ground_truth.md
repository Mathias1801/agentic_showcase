# Ground Truth — Injected Anomalies (NOT for agents)

Use this to write and grade the golden eval set. The agents should never see
this file — it exists so you can check whether the Analysis/Critic agents
actually found the right cause.

## 1. Q2 2025 margin drop — Home & Garden
- Supplier unit cost for all Home & Garden products rose 18%
  effective 2025-04-01.
- List price was not adjusted until 2025-05-15
  (partial pass-through, ~90% of the cost increase).
- Expected signature: gross_margin_pct for category = 'Home & Garden' drops
  sharply starting April 2025, recovers partially mid-May 2025.

## 2. Returns spike — Smart Speaker Gen2
- Product launched 2025-02-01.
- Defective batch caused return rate to jump to ~22%
  (vs ~3% baseline) between 2025-04-01 and
  2025-05-10.
- reason_code = 'DEFECTIVE_BATCH' in fact_returns isolates this.

## 3. Marketing ROI decline — Online channel, June 2025
- 'Summer Sale 2025' campaign increased Online spend ~2.6x
  from 2025-06-01 to 2025-06-30.
- Revenue-per-spend (ROI) for that period falls to roughly
  55% of the normal rate — spend scaled
  faster than incremental demand (diminishing returns), simulated by NOT scaling
  fact_sales revenue proportionally to the spend increase.

## 4. Regional disruption — UK & Ireland, March 2025
- Simulated logistics disruption from 2025-03-03 to
  2025-03-21.
- fact_sales transaction volume for region = 'UK & Ireland' suppressed to
  roughly 45% of normal during this window.
- fact_finance_ops shows elevated Logistics department opex
  (+1800/day) for department = 'Logistics'
  over the same window.

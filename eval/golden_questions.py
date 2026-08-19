"""
golden_questions.py

Defines the golden eval set: ~30 business questions against
data/business.db, split into four categories:

  A. simple_aggregation   — single-table or single-join, unambiguous (10)
  B. multi_join           — requires joining multiple fact tables (8)
  C. causal_ambiguous      — "why did X happen" questions tied to the
                              anomalies in docs/ground_truth.md (8)
  D. control_no_anomaly    — surface-level looks like it could have a
                              cause, but nothing was injected; correct
                              behavior is to NOT invent a root cause (4)

Each question carries the SQL needed to compute its own ground truth
directly from business.db, so the "correct answer" is derived from the
data itself rather than hand-typed and prone to drifting out of sync.

Grading against these questions should check three things, matching the
project's eval design:
  1. retrieval accuracy   — did the agent pull the right numbers/rows
  2. grounding accuracy   — does the final answer match what was retrieved
                             (no invented figures)
  3. LLM-as-judge score   — qualitative quality of the explanation, scored
                             by a DIFFERENT model than the one that
                             generated the answer

Category D exists specifically to catch hallucinated causality: a system
that always finds a "root cause" for every question (even ones with no
real anomaly) should score poorly here even if it sounds confident.

Run this file directly to compute and print/save all ground-truth answers:
    python eval/golden_questions.py
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "business.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "golden_answers.json"


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------
# Each question: id, category, difficulty, question, queries (named SQL to
# run for ground truth), answer_notes (what a correct answer must contain).

GOLDEN_QUESTIONS = [

    # ---- Category A: simple aggregation (easy, single fact) --------------
    {
        "id": "A01", "category": "simple_aggregation", "difficulty": "easy",
        "question": "What was total gross revenue in 2024?",
        "queries": {
            "total_revenue_2024": """
                SELECT ROUND(SUM(s.gross_revenue), 2) AS value
                FROM fact_sales s JOIN dim_date d ON s.date_id = d.date_id
                WHERE d.year = 2024
            """
        },
        "answer_notes": "Single number: total 2024 gross revenue.",
    },
    {
        "id": "A02", "category": "simple_aggregation", "difficulty": "easy",
        "question": "What was total gross revenue in 2025?",
        "queries": {
            "total_revenue_2025": """
                SELECT ROUND(SUM(s.gross_revenue), 2) AS value
                FROM fact_sales s JOIN dim_date d ON s.date_id = d.date_id
                WHERE d.year = 2025
            """
        },
        "answer_notes": "Single number: total 2025 gross revenue.",
    },
    {
        "id": "A03", "category": "simple_aggregation", "difficulty": "easy",
        "question": "How many units of 'Wireless Earbuds Pro' were sold in 2024?",
        "queries": {
            "units_sold": """
                SELECT SUM(s.quantity) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_product p ON s.product_id = p.product_id
                WHERE p.product_name = 'Wireless Earbuds Pro' AND d.year = 2024
            """
        },
        "answer_notes": "Single number: total units.",
    },
    {
        "id": "A04", "category": "simple_aggregation", "difficulty": "easy",
        "question": "What is the average discount percentage given across all 2025 sales?",
        "queries": {
            "avg_discount_pct": """
                SELECT ROUND(AVG(s.discount_pct) * 100, 2) AS value
                FROM fact_sales s JOIN dim_date d ON s.date_id = d.date_id
                WHERE d.year = 2025
            """
        },
        "answer_notes": "Single percentage.",
    },
    {
        "id": "A05", "category": "simple_aggregation", "difficulty": "easy",
        "question": "Which sales channel generated the most gross revenue in 2025?",
        "queries": {
            "revenue_by_channel": """
                SELECT c.channel_name, ROUND(SUM(s.gross_revenue), 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_channel c ON s.channel_id = c.channel_id
                WHERE d.year = 2025
                GROUP BY c.channel_name ORDER BY value DESC
            """
        },
        "answer_notes": "Name the top channel and its revenue figure.",
    },
    {
        "id": "A06", "category": "simple_aggregation", "difficulty": "easy",
        "question": "What was total marketing spend in 2025?",
        "queries": {
            "total_spend_2025": """
                SELECT ROUND(SUM(m.spend_amount), 2) AS value
                FROM fact_marketing_spend m JOIN dim_date d ON m.date_id = d.date_id
                WHERE d.year = 2025
            """
        },
        "answer_notes": "Single number.",
    },
    {
        "id": "A07", "category": "simple_aggregation", "difficulty": "easy",
        "question": "How many returns were recorded in 2025?",
        "queries": {
            "return_count_2025": """
                SELECT COUNT(*) AS value
                FROM fact_returns r JOIN dim_date d ON r.date_id = d.date_id
                WHERE d.year = 2025
            """
        },
        "answer_notes": "Single count of return records (not units).",
    },
    {
        "id": "A08", "category": "simple_aggregation", "difficulty": "easy",
        "question": "What is the average order quantity across all sales (both years)?",
        "queries": {
            "avg_quantity": "SELECT ROUND(AVG(quantity), 2) AS value FROM fact_sales"
        },
        "answer_notes": "Single number.",
    },
    {
        "id": "A09", "category": "simple_aggregation", "difficulty": "easy",
        "question": "Which customer segment had the highest total gross revenue in 2024?",
        "queries": {
            "revenue_by_segment": """
                SELECT seg.segment_name, ROUND(SUM(s.gross_revenue), 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_segment seg ON s.segment_id = seg.segment_id
                WHERE d.year = 2024
                GROUP BY seg.segment_name ORDER BY value DESC
            """
        },
        "answer_notes": "Name the top segment and its revenue figure.",
    },
    {
        "id": "A10", "category": "simple_aggregation", "difficulty": "easy",
        "question": "What was total Corporate Overhead opex spend in 2025?",
        "queries": {
            "overhead_2025": """
                SELECT ROUND(SUM(f.amount), 2) AS value
                FROM fact_finance_ops f JOIN dim_date d ON f.date_id = d.date_id
                WHERE d.year = 2025 AND f.department = 'Corporate Overhead'
            """
        },
        "answer_notes": "Single number.",
    },

    # ---- Category B: multi-join (moderate) --------------------------------
    {
        "id": "B01", "category": "multi_join", "difficulty": "medium",
        "question": "What was the gross margin percentage for the Electronics category "
                     "in 2024, and how did it compare to 2025?",
        "queries": {
            "margin_by_year": """
                SELECT d.year, ROUND(AVG(s.gross_margin_pct) * 100, 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_product p ON s.product_id = p.product_id
                WHERE p.category = 'Electronics'
                GROUP BY d.year
            """
        },
        "answer_notes": "Two percentages (2024 vs 2025) with the comparison. "
                         "Electronics has no injected anomaly, so expect the two "
                         "years to be close (no big swing).",
    },
    {
        "id": "B02", "category": "multi_join", "difficulty": "medium",
        "question": "Which region had the highest gross revenue in 2025, and what was its total?",
        "queries": {
            "revenue_by_region": """
                SELECT r.region_name, ROUND(SUM(s.gross_revenue), 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_region r ON s.region_id = r.region_id
                WHERE d.year = 2025
                GROUP BY r.region_name ORDER BY value DESC
            """
        },
        "answer_notes": "Name the top region and its revenue figure.",
    },
    {
        "id": "B03", "category": "multi_join", "difficulty": "medium",
        "question": "What was the return rate (as % of units sold) for the Electronics "
                     "category in 2025?",
        "queries": {
            "units_sold": """
                SELECT SUM(s.quantity) AS value
                FROM fact_sales s JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_product p ON s.product_id = p.product_id
                WHERE p.category = 'Electronics' AND d.year = 2025
            """,
            "units_returned": """
                SELECT SUM(r.quantity_returned) AS value
                FROM fact_returns r JOIN dim_date d ON r.date_id = d.date_id
                JOIN dim_product p ON r.product_id = p.product_id
                WHERE p.category = 'Electronics' AND d.year = 2025
            """,
        },
        "answer_notes": "Compute units_returned / units_sold as a percentage. "
                         "Elevated vs other categories because it includes the "
                         "Smart Speaker Gen2 defect period.",
    },
    {
        "id": "B04", "category": "multi_join", "difficulty": "medium",
        "question": "How did Wholesale channel gross margin percentage compare to "
                     "Online in 2025?",
        "queries": {
            "margin_by_channel": """
                SELECT c.channel_name, ROUND(AVG(s.gross_margin_pct) * 100, 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_channel c ON s.channel_id = c.channel_id
                WHERE d.year = 2025 AND c.channel_name IN ('Wholesale', 'Online')
                GROUP BY c.channel_name
            """
        },
        "answer_notes": "Two percentages. Margin is not modeled to differ "
                         "systematically by channel — expect them close.",
    },
    {
        "id": "B05", "category": "multi_join", "difficulty": "hard",
        "question": "What was the ratio of marketing spend to gross revenue, by "
                     "channel, in 2025?",
        "queries": {
            "spend_by_channel": """
                SELECT c.channel_name, ROUND(SUM(m.spend_amount), 2) AS spend
                FROM fact_marketing_spend m
                JOIN dim_date d ON m.date_id = d.date_id
                JOIN dim_channel c ON m.channel_id = c.channel_id
                WHERE d.year = 2025 GROUP BY c.channel_name
            """,
            "revenue_by_channel": """
                SELECT c.channel_name, ROUND(SUM(s.gross_revenue), 2) AS revenue
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_channel c ON s.channel_id = c.channel_id
                WHERE d.year = 2025 GROUP BY c.channel_name
            """,
        },
        "answer_notes": "Compute spend/revenue per channel from the two result sets. "
                         "Online will look elevated for the full year because of the "
                         "June spike (see C05/C06) — a good answer should note that "
                         "the annual Online figure is pulled up by June specifically.",
    },
    {
        "id": "B06", "category": "multi_join", "difficulty": "medium",
        "question": "Which product category had the lowest average gross margin "
                     "percentage in 2025?",
        "queries": {
            "margin_by_category": """
                SELECT p.category, ROUND(AVG(s.gross_margin_pct) * 100, 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_product p ON s.product_id = p.product_id
                WHERE d.year = 2025
                GROUP BY p.category ORDER BY value ASC
            """
        },
        "answer_notes": "Name the lowest category. If it's Home & Garden, a strong "
                         "answer notes this is pulled down by the Q2 cost-increase "
                         "period rather than being low all year.",
    },
    {
        "id": "B07", "category": "multi_join", "difficulty": "medium",
        "question": "What was total Logistics department opex in 2025, broken down "
                     "by quarter?",
        "queries": {
            "logistics_by_quarter": """
                SELECT d.quarter, ROUND(SUM(f.amount), 2) AS value
                FROM fact_finance_ops f JOIN dim_date d ON f.date_id = d.date_id
                WHERE d.year = 2025 AND f.department = 'Logistics'
                GROUP BY d.quarter ORDER BY d.quarter
            """
        },
        "answer_notes": "Four quarterly figures. Q1 should be slightly elevated "
                         "from the March disruption's extra opex.",
    },
    {
        "id": "B08", "category": "multi_join", "difficulty": "medium",
        "question": "Which product had the highest total returns (by quantity) in 2025?",
        "queries": {
            "returns_by_product": """
                SELECT p.product_name, SUM(r.quantity_returned) AS value
                FROM fact_returns r
                JOIN dim_date d ON r.date_id = d.date_id
                JOIN dim_product p ON r.product_id = p.product_id
                WHERE d.year = 2025
                GROUP BY p.product_name ORDER BY value DESC LIMIT 5
            """
        },
        "answer_notes": "Name the top product (expect Smart Speaker Gen2) and its "
                         "return quantity.",
    },

    # ---- Category C: causal / ambiguous (tied to injected anomalies) -----
    {
        "id": "C01", "category": "causal_ambiguous", "difficulty": "hard",
        "question": "Why did gross margin percentage drop for Home & Garden products "
                     "in Q2 2025?",
        "queries": {
            "margin_by_month": """
                SELECT d.month, ROUND(AVG(s.gross_margin_pct) * 100, 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_product p ON s.product_id = p.product_id
                WHERE p.category = 'Home & Garden' AND d.year = 2025
                GROUP BY d.month ORDER BY d.month
            """,
        },
        "answer_notes": "Root cause (see docs/ground_truth.md, anomaly 1): supplier "
                         "unit cost for Home & Garden rose ~18% on 2025-04-01; list "
                         "price wasn't adjusted until 2025-05-15. Answer must name "
                         "the category, the timing, and cost-vs-price lag as the "
                         "mechanism — not just restate that margin dropped.",
    },
    {
        "id": "C02", "category": "causal_ambiguous", "difficulty": "hard",
        "question": "Home & Garden margin recovered somewhat in May 2025 but not "
                     "fully back to Q1 levels — why?",
        "queries": {
            "margin_by_month": """
                SELECT d.month, ROUND(AVG(s.gross_margin_pct) * 100, 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_product p ON s.product_id = p.product_id
                WHERE p.category = 'Home & Garden' AND d.year = 2025
                GROUP BY d.month ORDER BY d.month
            """,
        },
        "answer_notes": "The 2025-05-15 price increase only covered ~90% of the "
                         "cost increase (partial pass-through), so margin should "
                         "land close to but below the Q1 baseline from June onward, "
                         "not fully recover. Tests whether the agent notices a "
                         "partial vs full recovery.",
    },
    {
        "id": "C03", "category": "causal_ambiguous", "difficulty": "hard",
        "question": "What's driving the spike in returns for Smart Speaker Gen2 in "
                     "spring 2025?",
        "queries": {
            "returns_by_month": """
                SELECT d.month, r.reason_code, SUM(r.quantity_returned) AS value
                FROM fact_returns r
                JOIN dim_date d ON r.date_id = d.date_id
                JOIN dim_product p ON r.product_id = p.product_id
                WHERE p.product_name = 'Smart Speaker Gen2' AND d.year = 2025
                GROUP BY d.month, r.reason_code ORDER BY d.month
            """,
        },
        "answer_notes": "Root cause (anomaly 2): a defective batch, reason_code "
                         "'DEFECTIVE_BATCH', concentrated 2025-04-01 to 2025-05-10. "
                         "Answer must name the product, the reason code, and the "
                         "window — not just 'returns increased'.",
    },
    {
        "id": "C04", "category": "causal_ambiguous", "difficulty": "medium",
        "question": "Is the Smart Speaker Gen2 returns issue isolated to one region, "
                     "or does it appear across all regions?",
        "queries": {
            "returns_by_region": """
                SELECT reg.region_name, SUM(r.quantity_returned) AS value
                FROM fact_returns r
                JOIN dim_date d ON r.date_id = d.date_id
                JOIN dim_product p ON r.product_id = p.product_id
                JOIN dim_region reg ON r.region_id = reg.region_id
                WHERE p.product_name = 'Smart Speaker Gen2' AND d.year = 2025
                  AND d.date BETWEEN '2025-04-01' AND '2025-05-10'
                GROUP BY reg.region_name ORDER BY value DESC
            """,
        },
        "answer_notes": "The defect is a manufacturing/product issue, not "
                         "region-specific, so it should appear across all regions "
                         "roughly proportional to their normal sales volume — a good "
                         "answer says it's a product-wide (not regional) issue.",
    },
    {
        "id": "C05", "category": "causal_ambiguous", "difficulty": "hard",
        "question": "Why did Online channel marketing efficiency (revenue per dollar "
                     "spent) decline in June 2025?",
        "queries": {
            "spend_and_revenue_by_month": """
                SELECT d.month,
                  (SELECT ROUND(SUM(m.spend_amount),2) FROM fact_marketing_spend m
                     JOIN dim_channel c ON m.channel_id=c.channel_id
                     WHERE c.channel_name='Online' AND m.date_id IN
                       (SELECT date_id FROM dim_date WHERE year=2025 AND month=d.month)
                  ) AS online_spend,
                  (SELECT ROUND(SUM(s.gross_revenue),2) FROM fact_sales s
                     JOIN dim_channel c ON s.channel_id=c.channel_id
                     WHERE c.channel_name='Online' AND s.date_id IN
                       (SELECT date_id FROM dim_date WHERE year=2025 AND month=d.month)
                  ) AS online_revenue
                FROM (SELECT DISTINCT month FROM dim_date WHERE year=2025) d
                ORDER BY d.month
            """,
        },
        "answer_notes": "Root cause (anomaly 3): the 'Summer Sale 2025' campaign "
                         "raised Online spend ~2.6x in June without a proportional "
                         "revenue lift — diminishing returns from scaling spend into "
                         "an already-saturated audience. Answer should name the "
                         "campaign and quantify spend vs revenue growth, not just "
                         "say 'spend went up'.",
    },
    {
        "id": "C06", "category": "causal_ambiguous", "difficulty": "medium",
        "question": "Was the June 2025 Online spend increase associated with a "
                     "proportional increase in sales volume?",
        "queries": {
            "same_as_C05": "see C05 spend_and_revenue_by_month query",
        },
        "answer_notes": "No — spend roughly tripled while Online revenue grew only "
                         "modestly in line with the rest of the year's trend. A "
                         "correct answer explicitly says the increase was "
                         "disproportionate (not proportional).",
    },
    {
        "id": "C07", "category": "causal_ambiguous", "difficulty": "hard",
        "question": "Why did UK & Ireland sales volume drop sharply in March 2025?",
        "queries": {
            "daily_sales_count": """
                SELECT d.date, COUNT(*) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_region r ON s.region_id = r.region_id
                WHERE r.region_name = 'UK & Ireland'
                  AND d.date BETWEEN '2025-02-15' AND '2025-04-05'
                GROUP BY d.date ORDER BY d.date
            """,
        },
        "answer_notes": "Root cause (anomaly 4): a simulated logistics disruption "
                         "in that region from 2025-03-03 to 2025-03-21. Answer "
                         "should name the region and the specific date window, not "
                         "just note that volume fell.",
    },
    {
        "id": "C08", "category": "causal_ambiguous", "difficulty": "medium",
        "question": "What financial impact did the March 2025 UK & Ireland "
                     "disruption have beyond lost sales?",
        "queries": {
            "logistics_opex_around_disruption": """
                SELECT d.date, f.amount
                FROM fact_finance_ops f
                JOIN dim_date d ON f.date_id = d.date_id
                WHERE f.department = 'Logistics'
                  AND d.date BETWEEN '2025-02-20' AND '2025-03-28'
                ORDER BY d.date
            """,
        },
        "answer_notes": "Logistics department opex was elevated (+$1,800/day) for "
                         "the disruption window. A correct answer connects the "
                         "opex bump to the same date range as the sales drop, "
                         "i.e. it's the same underlying event, not two separate ones.",
    },

    # ---- Category D: control questions — no injected anomaly -------------
    {
        "id": "D01", "category": "control_no_anomaly", "difficulty": "medium",
        "question": "Why did Apparel category revenue vary month to month in 2025?",
        "queries": {
            "revenue_by_month": """
                SELECT d.month, ROUND(SUM(s.gross_revenue), 2) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_product p ON s.product_id = p.product_id
                WHERE p.category = 'Apparel' AND d.year = 2025
                GROUP BY d.month ORDER BY d.month
            """,
        },
        "answer_notes": "TRAP QUESTION: no anomaly is injected for Apparel. "
                         "Month-to-month variation here is ordinary random noise "
                         "(daily transaction counts are Poisson-distributed with "
                         "weekday/weekend/holiday effects only). A good answer "
                         "attributes this to normal variation and explicitly does "
                         "NOT invent a specific root cause. A system that produces "
                         "a confident causal story here should be marked down.",
    },
    {
        "id": "D02", "category": "control_no_anomaly", "difficulty": "medium",
        "question": "What explains the difference in gross margin between Consumer "
                     "and Enterprise segments?",
        "queries": {
            "margin_by_segment": """
                SELECT seg.segment_name, ROUND(AVG(s.gross_margin_pct) * 100, 2) AS value
                FROM fact_sales s
                JOIN dim_segment seg ON s.segment_id = seg.segment_id
                GROUP BY seg.segment_name
            """,
        },
        "answer_notes": "TRAP QUESTION: margin is not modeled to depend on "
                         "customer segment at all in the generator, so any "
                         "difference in the numbers is sampling noise. Correct "
                         "answer notes the difference is negligible/not "
                         "meaningful, rather than inventing a segment-based "
                         "explanation (e.g. 'Enterprise gets better pricing').",
    },
    {
        "id": "D03", "category": "control_no_anomaly", "difficulty": "medium",
        "question": "Why did Wholesale channel marketing spend increase over the "
                     "course of 2025?",
        "queries": {
            "spend_by_month": """
                SELECT d.month, ROUND(SUM(m.spend_amount), 2) AS value
                FROM fact_marketing_spend m
                JOIN dim_date d ON m.date_id = d.date_id
                JOIN dim_channel c ON m.channel_id = c.channel_id
                WHERE c.channel_name = 'Wholesale' AND d.year = 2025
                GROUP BY d.month ORDER BY d.month
            """,
        },
        "answer_notes": "TRAP QUESTION: the premise may be false. Wholesale "
                         "spend is generated as a fixed base +/- random noise "
                         "with no growth trend or campaign. A good answer checks "
                         "the premise, reports that there is no real upward trend "
                         "(just noise), and does not fabricate a cause for a "
                         "trend that isn't really there.",
    },
    {
        "id": "D04", "category": "control_no_anomaly", "difficulty": "medium",
        "question": "Did the Southern Europe region experience any unusual sales "
                     "disruption in 2025?",
        "queries": {
            "daily_sales_count": """
                SELECT d.month, COUNT(*) AS value
                FROM fact_sales s
                JOIN dim_date d ON s.date_id = d.date_id
                JOIN dim_region r ON s.region_id = r.region_id
                WHERE r.region_name = 'Southern Europe' AND d.year = 2025
                GROUP BY d.month ORDER BY d.month
            """,
        },
        "answer_notes": "TRAP QUESTION: no anomaly is assigned to Southern "
                         "Europe (only UK & Ireland has the logistics disruption). "
                         "Correct answer is 'no, nothing unusual' — tests whether "
                         "the agent falsely generalizes the real UK & Ireland "
                         "disruption to a region it didn't affect.",
    },
]


# ---------------------------------------------------------------------------
# Compute ground truth
# ---------------------------------------------------------------------------

def compute_all(db_path: Path = DB_PATH) -> list:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    results = []

    for q in GOLDEN_QUESTIONS:
        computed = {}
        for name, sql in q["queries"].items():
            if sql.strip().lower().startswith("see "):
                computed[name] = sql  # cross-reference note, not a real query
                continue
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
            computed[name] = rows

        results.append({
            "id": q["id"],
            "category": q["category"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "answer_notes": q["answer_notes"],
            "computed_ground_truth": computed,
        })

    conn.close()
    return results


def main():
    results = compute_all()
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"Computed ground truth for {len(results)} questions -> {OUTPUT_PATH}")

    by_category = {}
    for q in GOLDEN_QUESTIONS:
        by_category.setdefault(q["category"], 0)
        by_category[q["category"]] += 1
    print("Breakdown by category:")
    for cat, n in by_category.items():
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    main()

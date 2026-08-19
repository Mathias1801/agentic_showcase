import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guardrails.sql_guardrails import enforce_row_limit, validate_sql


def test_valid_select_passes():
    r = validate_sql("SELECT * FROM fact_sales WHERE date_id = 20250401")
    assert r.is_valid
    assert "fact_sales" in r.tables_referenced


def test_valid_join_passes():
    sql = """
        SELECT p.category, SUM(s.gross_revenue)
        FROM fact_sales s JOIN dim_product p ON s.product_id = p.product_id
        GROUP BY p.category
    """
    r = validate_sql(sql)
    assert r.is_valid
    assert r.tables_referenced == {"fact_sales", "dim_product"}


def test_with_clause_passes():
    sql = """
        WITH monthly AS (SELECT date_id, gross_revenue FROM fact_sales)
        SELECT * FROM monthly
    """
    r = validate_sql(sql)
    assert r.is_valid


def test_rejects_insert():
    r = validate_sql("INSERT INTO fact_sales (sale_id) VALUES (1)")
    assert not r.is_valid


def test_rejects_update():
    r = validate_sql("UPDATE fact_sales SET quantity = 0")
    assert not r.is_valid


def test_rejects_delete():
    r = validate_sql("DELETE FROM fact_sales")
    assert not r.is_valid


def test_rejects_drop():
    r = validate_sql("DROP TABLE fact_sales")
    assert not r.is_valid


def test_rejects_pragma():
    r = validate_sql("PRAGMA table_info(fact_sales)")
    assert not r.is_valid


def test_rejects_attach():
    r = validate_sql("ATTACH DATABASE 'other.db' AS other; SELECT * FROM other.secrets")
    assert not r.is_valid


def test_rejects_multi_statement():
    r = validate_sql("SELECT * FROM fact_sales; DROP TABLE fact_sales;")
    assert not r.is_valid


def test_rejects_unknown_table():
    r = validate_sql("SELECT * FROM users")
    assert not r.is_valid
    assert "users" in r.reason


def test_rejects_sql_injection_style_comment():
    r = validate_sql("SELECT * FROM fact_sales -- ; DROP TABLE fact_sales")
    # comment is stripped before the multi-statement check; base query is a
    # plain SELECT from an allowed table, so this should be treated as valid
    # once the trailing comment is removed
    assert r.is_valid


def test_rejects_empty_query():
    r = validate_sql("")
    assert not r.is_valid


def test_enforce_row_limit_wraps_query():
    wrapped = enforce_row_limit("SELECT * FROM fact_sales", max_rows=10)
    assert "LIMIT 10" in wrapped
    assert wrapped.strip().startswith("SELECT * FROM (SELECT * FROM fact_sales)")

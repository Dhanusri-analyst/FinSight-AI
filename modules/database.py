import sqlite3
from pathlib import Path

# Get the FinSight AI project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Store the database inside the project's database folder
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)

DB_NAME = DB_DIR / "finsight.db"


def connect():
    return sqlite3.connect(str(DB_NAME), timeout=30)


def create_table():
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT,
                date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def add_expense(description, amount, category="Unknown"):
    with connect() as conn:
        conn.execute("""
            INSERT INTO expenses (description, amount, category)
            VALUES (?, ?, ?)
        """, (description, amount, category))


def get_expenses():
    with connect() as conn:
        cursor = conn.execute("""
            SELECT * FROM expenses
            ORDER BY date DESC
        """)
        return cursor.fetchall()


def total_spending():
    with connect() as conn:
        cursor = conn.execute("""
            SELECT COALESCE(SUM(amount), 0)
            FROM expenses
        """)
        return cursor.fetchone()[0]


def monthly_spending(year, month):
    with connect() as conn:
        cursor = conn.execute("""
            SELECT COALESCE(SUM(amount), 0)
            FROM expenses
            WHERE strftime('%Y', date) = ?
              AND strftime('%m', date) = ?
        """, (str(year), f"{month:02d}"))
        return cursor.fetchone()[0]


def delete_expense(expense_id):
    with connect() as conn:
        conn.execute(
            "DELETE FROM expenses WHERE id = ?",
            (expense_id,)
        )


def update_expense(expense_id, description, amount, category):
    with connect() as conn:
        conn.execute("""
            UPDATE expenses
            SET description = ?,
                amount = ?,
                category = ?
            WHERE id = ?
        """, (description, amount, category, expense_id))


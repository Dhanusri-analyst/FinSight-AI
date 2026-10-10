import sqlite3

DB_NAME = "database/finsight.db"


def connect():
    return sqlite3.connect(DB_NAME)


def create_table():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS expenses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            category TEXT,
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def add_expense(description, amount, category="Unknown"):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO expenses(description, amount, category)
        VALUES (?, ?, ?)
    """, (description, amount, category))

    conn.commit()
    conn.close()


def get_expenses():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM expenses
        ORDER BY date DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows


def total_spending():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0)
        FROM expenses
    """)

    total = cursor.fetchone()[0]

    conn.close()

    return total


def monthly_spending(year, month):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COALESCE(SUM(amount), 0)
        FROM expenses
        WHERE strftime('%Y', date) = ?
        AND strftime('%m', date) = ?
    """, (
        str(year),
        f"{month:02d}"
    ))

    total = cursor.fetchone()[0]

    conn.close()

    return total


def delete_expense(expense_id):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM expenses WHERE id = ?",
        (expense_id,)
    )

    conn.commit()
    conn.close()


def update_expense(expense_id, description, amount, category):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE expenses
        SET description = ?,
            amount = ?,
            category = ?
        WHERE id = ?
    """, (
        description,
        amount,
        category,
        expense_id
    ))

    conn.commit()
    conn.close()

import sqlite3


def create_sqlite(path):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # ステップテーブルの作成
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT
    )
    """
    )

    # 位置テーブルの作成
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        step_id INTEGER,
        x INTEGER,
        y INTEGER,
        FOREIGN KEY(step_id) REFERENCES steps(id)
    )
    """
    )

    conn.commit()
    conn.close()


def save_sqlite(path, pos):
    conn = sqlite3.connect(path)
    c = conn.cursor()

    # 新しいステップを追加
    c.execute("INSERT INTO steps DEFAULT VALUES")
    step_id = c.lastrowid

    # 位置データを追加
    # for y, x in zip(pos[:, 0], pos[:, 1]):
    #     c.execute(
    #         "INSERT INTO positions (step_id, x, y) VALUES (?, ?, ?)",
    #         (step_id, int(x), int(y)),
    #     )
    insert_data = [
        (step_id, int(x), int(y)) for y, x in zip(pos[:, 0], pos[:, 1])
    ]
    c.executemany(
        "INSERT INTO positions (step_id, x, y) VALUES (?, ?, ?)", insert_data
    )

    conn.commit()
    conn.close()

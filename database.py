import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path


class Repository:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS areas (id TEXT PRIMARY KEY, payload TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS observations (id INTEGER PRIMARY KEY, area_id TEXT NOT NULL REFERENCES areas(id), period_start TEXT NOT NULL, period_end TEXT NOT NULL, daily INTEGER NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS observation_area_period ON observations(area_id, period_start, period_end);
                CREATE TABLE IF NOT EXISTS predictions (id TEXT PRIMARY KEY, area_id TEXT NOT NULL REFERENCES areas(id), issued_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS prediction_area_date ON predictions(area_id, issued_at);
                CREATE TABLE IF NOT EXISTS outlooks (id TEXT PRIMARY KEY, area_id TEXT NOT NULL REFERENCES areas(id), created_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS outlook_area_created ON outlooks(area_id, created_at);
                CREATE TABLE IF NOT EXISTS prediction_requests (id TEXT PRIMARY KEY, area_id TEXT, requested_at TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS request_area_created ON prediction_requests(area_id, requested_at);
            """)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def import_dataset(self, dataset):
        with self.connection() as connection:
            connection.execute("UPDATE areas SET active = 0")
            connection.executemany("INSERT INTO areas (id, payload, active) VALUES (?, ?, 1) ON CONFLICT(id) DO UPDATE SET payload = excluded.payload, active = 1", [(area["id"], json.dumps(area)) for area in dataset["areas"]])
            connection.execute("DELETE FROM observations")
            for daily, records in [(0, dataset["observations"]), (1, dataset["daily_observations"])]:
                rows = [(record["area_id"], record["start"], record["end"], daily, json.dumps(record)) for record in records if not record["grouped"]]
                connection.executemany("INSERT INTO observations (area_id, period_start, period_end, daily, payload) VALUES (?, ?, ?, ?, ?)", rows)

    def areas(self):
        with self.connection() as connection:
            rows = connection.execute("SELECT payload FROM areas WHERE active = 1").fetchall()
        return sorted([json.loads(row[0]) for row in rows], key=lambda area: (-area["daily_records"], area["name"]))

    def area(self, identifier):
        with self.connection() as connection:
            row = connection.execute("SELECT payload FROM areas WHERE id = ? AND active = 1", (identifier,)).fetchone()
        return json.loads(row[0]) if row else None

    def observations(self, identifier, start, end, daily=False):
        with self.connection() as connection:
            rows = connection.execute("SELECT payload FROM observations WHERE area_id = ? AND daily = ? AND period_start <= ? AND period_end >= ? ORDER BY period_start", (identifier, int(daily), end, start)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_prediction(self, prediction):
        with self.connection() as connection:
            connection.execute("INSERT OR IGNORE INTO predictions (id, area_id, issued_at, payload) VALUES (?, ?, ?, ?)", (prediction["id"], prediction["area_id"], prediction["issued_at"], json.dumps(prediction, allow_nan=False)))

    def predictions(self, identifier=None):
        with self.connection() as connection:
            if identifier:
                rows = connection.execute("SELECT payload FROM predictions WHERE area_id = ? ORDER BY issued_at DESC, id DESC LIMIT 100", (identifier,)).fetchall()
            else:
                rows = connection.execute("SELECT payload FROM predictions ORDER BY issued_at DESC, id DESC LIMIT 100").fetchall()
        return [json.loads(row[0]) for row in rows]

    def health(self):
        with self.connection() as connection:
            return connection.execute("SELECT 1").fetchone()[0] == 1

    def save_outlook(self, outlook):
        with self.connection() as connection:
            connection.execute("INSERT OR IGNORE INTO outlooks (id, area_id, created_at, payload) VALUES (?, ?, ?, ?)", (outlook["id"], outlook["area"]["id"], outlook["generated_at"], json.dumps(outlook, allow_nan=False)))

    def save_prediction_request(self, record):
        with self.connection() as connection:
            connection.execute("INSERT INTO prediction_requests (id, area_id, requested_at, payload) VALUES (?, ?, ?, ?)", (record["id"], record.get("area_id"), record["requested_at"], json.dumps(record, allow_nan=False)))

    def prediction_requests(self, identifier=None):
        with self.connection() as connection:
            if identifier:
                rows = connection.execute("SELECT payload FROM prediction_requests WHERE area_id = ? ORDER BY requested_at DESC, id DESC LIMIT 100", (identifier,)).fetchall()
            else:
                rows = connection.execute("SELECT payload FROM prediction_requests ORDER BY requested_at DESC, id DESC LIMIT 100").fetchall()
        return [json.loads(row[0]) for row in rows]

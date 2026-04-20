"""
PostgreSQL helper module for serverless applications.
"""

import psycopg
import os
import json as _json


def get_connection():
    DB_HOST     = os.environ.get("DB_HOST")
    DB_NAME     = os.environ.get("DB_NAME")
    DB_PASSWORD = os.environ.get("DB_PASSWORD")
    DB_PORT     = os.environ.get("DB_PORT")
    DB_USERNAME = os.environ.get("DB_USERNAME")

    if not all([DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USERNAME]):
        print("DB connection error: Missing one or more environment variables.")
        return None

    try:
        conn = psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USERNAME,
            password=DB_PASSWORD
        )
        return conn
    except Exception as e:
        print(f"DB connection error: {e}")
        return None


def execute_query(query):
    conn = get_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            records = cursor.fetchall()
            return records
    except Exception as e:
        print(f"Query execution error: {e}")
        return []
    finally:
        conn.close()


def insert_rppg_result(user_id, input_data, production_result, reference_result):
    conn = get_connection()
    if conn is None:
        return None

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO rppg.rppg_models
                    (user_id, input_data, production_result, reference_result)
                VALUES (%s, %s, %s, %s)
                RETURNING reading_id, created_at
            """, (
                str(user_id),
                _json.dumps(input_data),
                _json.dumps(production_result),
                _json.dumps(reference_result)
            ))
            conn.commit()
            row = cursor.fetchone()
            return {"reading_id": str(row[0]), "created_at": str(row[1])}
    except Exception as e:
        print(f"Insert error: {e}")
        return None
    finally:
        conn.close()


def get_rppg_results(user_id=None):
    conn = get_connection()
    if conn is None:
        return []

    try:
        with conn.cursor() as cursor:
            if user_id:
                cursor.execute("""
                    SELECT reading_id, user_id, input_data,
                           production_result, reference_result,
                           created_at, updated_at
                    FROM rppg.rppg_models
                    WHERE user_id = %s AND deleted_at IS NULL
                    ORDER BY created_at DESC
                """, (str(user_id),))
            else:
                cursor.execute("""
                    SELECT reading_id, user_id, input_data,
                           production_result, reference_result,
                           created_at, updated_at
                    FROM rppg.rppg_models
                    WHERE deleted_at IS NULL
                    ORDER BY created_at DESC
                """)

            rows = cursor.fetchall()
            return [
                {
                    "reading_id":        str(r[0]),
                    "user_id":           str(r[1]),
                    "input_data":        r[2],
                    "production_result": r[3],
                    "reference_result":  r[4],
                    "created_at":        str(r[5]),
                    "updated_at":        str(r[6]),
                }
                for r in rows
            ]
    except Exception as e:
        print(f"Fetch error: {e}")
        return []
    finally:
        conn.close()

import sqlite3
from pathlib import Path


# ==========================================
# DATABASE CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "facesort.db"


# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_connection(
    db_path: str | Path = DATABASE_PATH
) -> sqlite3.Connection:
    """Create a connection to the local FaceSort AI database."""

    db_path = Path(db_path)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.row_factory = sqlite3.Row

    return connection


# ==========================================
# IDENTITY DECISIONS TABLE
# ==========================================

def _create_identity_decisions_table(
    connection: sqlite3.Connection,
    table_name: str = "identity_decisions"
) -> None:
    """
    Create the identity decisions table.

    Supported candidate types:
    - GROUP_GROUP
    - UNMATCHED_FACE_GROUP
    - UNMATCHED_FACE_PAIR
    """

    if table_name not in {
        "identity_decisions",
        "identity_decisions_new"
    }:
        raise ValueError("Invalid identity decisions table name.")

    connection.execute(f"""
        CREATE TABLE {table_name} (

            decision_id INTEGER PRIMARY KEY AUTOINCREMENT,

            candidate_type TEXT NOT NULL
                CHECK (
                    candidate_type IN (
                        'GROUP_GROUP',
                        'UNMATCHED_FACE_GROUP',
                        'UNMATCHED_FACE_PAIR'
                    )
                ),

            person_a_id TEXT,

            person_b_id TEXT,

            face_record_id INTEGER,

            face_record_id_b INTEGER,

            decision TEXT NOT NULL
                CHECK (
                    decision IN (
                        'SAME_PERSON',
                        'DIFFERENT_PEOPLE',
                        'SKIPPED'
                    )
                ),

            maximum_similarity REAL,

            average_similarity REAL,

            minimum_similarity REAL,

            decided_at TEXT NOT NULL
                DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (person_a_id)
                REFERENCES people(person_id),

            FOREIGN KEY (person_b_id)
                REFERENCES people(person_id),

            FOREIGN KEY (face_record_id)
                REFERENCES face_records(face_id),

            FOREIGN KEY (face_record_id_b)
                REFERENCES face_records(face_id),

            CHECK (
                (
                    candidate_type = 'GROUP_GROUP'
                    AND person_a_id IS NOT NULL
                    AND person_b_id IS NOT NULL
                    AND person_a_id != person_b_id
                    AND face_record_id IS NULL
                    AND face_record_id_b IS NULL
                )
                OR
                (
                    candidate_type = 'UNMATCHED_FACE_GROUP'
                    AND person_a_id IS NOT NULL
                    AND person_b_id IS NULL
                    AND face_record_id IS NOT NULL
                    AND face_record_id_b IS NULL
                )
                OR
                (
                    candidate_type = 'UNMATCHED_FACE_PAIR'
                    AND person_a_id IS NULL
                    AND person_b_id IS NULL
                    AND face_record_id IS NOT NULL
                    AND face_record_id_b IS NOT NULL
                    AND face_record_id != face_record_id_b
                )
            )
        )
    """)


# ==========================================
# DATABASE MIGRATION
# ==========================================

def _migrate_identity_decisions(
    connection: sqlite3.Connection
) -> None:
    """
    Extend the identity decisions table while preserving
    all existing decisions.

    The migration adds:
    - UNMATCHED_FACE_PAIR candidate type
    - face_record_id_b column
    - Nullable person IDs for face-pair decisions
    """

    row = connection.execute("""
        SELECT sql
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'identity_decisions'
    """).fetchone()

    if row is None:
        _create_identity_decisions_table(connection)
        return

    existing_sql = row["sql"] or ""

    columns = {
        item["name"]
        for item in connection.execute(
            "PRAGMA table_info(identity_decisions)"
        ).fetchall()
    }

    migration_required = (
        "UNMATCHED_FACE_PAIR" not in existing_sql
        or "face_record_id_b" not in columns
    )

    if not migration_required:
        return

    # Rebuild the table inside a transaction.
    # Existing decisions are copied before the old table is dropped.
    connection.execute("BEGIN IMMEDIATE")

    try:

        _create_identity_decisions_table(
            connection,
            "identity_decisions_new"
        )

        connection.execute("""
            INSERT INTO identity_decisions_new (
                decision_id,
                candidate_type,
                person_a_id,
                person_b_id,
                face_record_id,
                face_record_id_b,
                decision,
                maximum_similarity,
                average_similarity,
                minimum_similarity,
                decided_at
            )
            SELECT
                decision_id,
                candidate_type,
                person_a_id,
                person_b_id,
                face_record_id,
                NULL,
                decision,
                maximum_similarity,
                average_similarity,
                minimum_similarity,
                decided_at
            FROM identity_decisions
        """)

        connection.execute("""
            DROP TABLE identity_decisions
        """)

        connection.execute("""
            ALTER TABLE identity_decisions_new
            RENAME TO identity_decisions
        """)

        connection.commit()

    except Exception:
        connection.rollback()
        raise


# ==========================================
# INDEXES
# ==========================================

def _create_indexes(
    connection: sqlite3.Connection
) -> None:

    connection.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_face_records_person
        ON face_records(person_id)
    """)

    connection.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_identity_decisions_person_a
        ON identity_decisions(person_a_id)
    """)

    connection.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_identity_decisions_person_b
        ON identity_decisions(person_b_id)
    """)

    connection.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_identity_decisions_face_a
        ON identity_decisions(face_record_id)
    """)

    connection.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_identity_decisions_face_b
        ON identity_decisions(face_record_id_b)
    """)


# ==========================================
# DATABASE INITIALIZATION
# ==========================================

def initialize_database(
    db_path: str | Path = DATABASE_PATH
) -> Path:
    """
    Create or migrate the FaceSort AI database.

    Existing people, face records and identity decisions
    are preserved.
    """

    connection = get_connection(db_path)

    try:

        with connection:

            # ----------------------------------
            # 1. PEOPLE
            # ----------------------------------

            connection.execute("""
                CREATE TABLE IF NOT EXISTS people (

                    person_id TEXT PRIMARY KEY,

                    display_name TEXT,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ----------------------------------
            # 2. FACE RECORDS
            # ----------------------------------

            connection.execute("""
                CREATE TABLE IF NOT EXISTS face_records (

                    face_id INTEGER PRIMARY KEY AUTOINCREMENT,

                    image_path TEXT NOT NULL,

                    face_number INTEGER NOT NULL,

                    person_id TEXT,

                    embedding BLOB,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (person_id)
                        REFERENCES people(person_id),

                    UNIQUE (image_path, face_number)
                )
            """)

        # Migration manages its own transaction.
        _migrate_identity_decisions(connection)

        with connection:

            _create_indexes(connection)

        return Path(db_path)

    finally:
        connection.close()

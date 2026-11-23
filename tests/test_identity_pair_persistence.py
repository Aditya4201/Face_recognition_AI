
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from src.database import initialize_database
from src.identity_repository import (
    create_person,
    save_face_record,
    save_identity_decision,
    get_identity_decisions
)


def test_fresh_database():

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "fresh_test.db"

        initialize_database(db_path)

        face_a = save_face_record(
            image_path=Path(temp_dir) / "photo_a.jpg",
            face_number=1,
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            db_path=db_path
        )

        face_b = save_face_record(
            image_path=Path(temp_dir) / "photo_b.jpg",
            face_number=1,
            embedding=np.array([0.2, 0.3, 0.4], dtype=np.float32),
            db_path=db_path
        )

        decision_id = save_identity_decision(
            candidate_type="UNMATCHED_FACE_PAIR",
            person_a_id=None,
            person_b_id=None,
            face_record_id=face_a,
            face_record_id_b=face_b,
            decision="SAME_PERSON",
            maximum_similarity=0.72,
            average_similarity=0.72,
            minimum_similarity=0.72,
            db_path=db_path
        )

        decisions = get_identity_decisions(db_path=db_path)

        assert len(decisions) == 1
        assert decisions[0]["decision_id"] == decision_id
        assert decisions[0]["candidate_type"] == "UNMATCHED_FACE_PAIR"
        assert decisions[0]["decision"] == "SAME_PERSON"

        assert {
            decisions[0]["face_record_id"],
            decisions[0]["face_record_id_b"]
        } == {face_a, face_b}

        print("PASS: Fresh database and face-pair persistence.")


def test_legacy_database_migration():

    with tempfile.TemporaryDirectory() as temp_dir:

        db_path = Path(temp_dir) / "legacy_test.db"

        connection = sqlite3.connect(db_path)

        connection.execute("PRAGMA foreign_keys = ON")

        # Create the original people table.
        connection.execute("""
            CREATE TABLE people (
                person_id TEXT PRIMARY KEY,
                display_name TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create the original face records table.
        connection.execute("""
            CREATE TABLE face_records (
                face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                face_number INTEGER NOT NULL,
                person_id TEXT,
                embedding BLOB,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES people(person_id),
                UNIQUE (image_path, face_number)
            )
        """)

        # Create the original identity decisions table.
        connection.execute("""
            CREATE TABLE identity_decisions (

                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,

                candidate_type TEXT NOT NULL
                    CHECK (
                        candidate_type IN (
                            'GROUP_GROUP',
                            'UNMATCHED_FACE_GROUP'
                        )
                    ),

                person_a_id TEXT NOT NULL,
                person_b_id TEXT,
                face_record_id INTEGER,

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

                decided_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (person_a_id) REFERENCES people(person_id),
                FOREIGN KEY (person_b_id) REFERENCES people(person_id),
                FOREIGN KEY (face_record_id) REFERENCES face_records(face_id),

                CHECK (
                    (
                        candidate_type = 'GROUP_GROUP'
                        AND person_b_id IS NOT NULL
                        AND face_record_id IS NULL
                    )
                    OR
                    (
                        candidate_type = 'UNMATCHED_FACE_GROUP'
                        AND person_b_id IS NULL
                        AND face_record_id IS NOT NULL
                    )
                )
            )
        """)

        # Insert existing people.
        connection.execute(
            "INSERT INTO people (person_id) VALUES (?)",
            ("Person_001",)
        )

        connection.execute(
            "INSERT INTO people (person_id) VALUES (?)",
            ("Person_002",)
        )

        # Insert an existing identity decision.
        connection.execute("""
            INSERT INTO identity_decisions (
                candidate_type,
                person_a_id,
                person_b_id,
                decision,
                maximum_similarity,
                average_similarity,
                minimum_similarity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "GROUP_GROUP",
            "Person_001",
            "Person_002",
            "DIFFERENT_PEOPLE",
            0.42,
            0.38,
            0.31
        ))

        connection.commit()
        connection.close()

        # Run the migration.
        initialize_database(db_path)

        decisions = get_identity_decisions(db_path=db_path)

        assert len(decisions) == 1
        assert decisions[0]["candidate_type"] == "GROUP_GROUP"
        assert decisions[0]["decision"] == "DIFFERENT_PEOPLE"

        # Confirm the new schema supports face pairs.
        connection = sqlite3.connect(db_path)

        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(identity_decisions)"
            ).fetchall()
        }

        connection.close()

        assert "face_record_id_b" in columns

        print("PASS: Legacy database migration.")
        print("PASS: Existing identity decision preserved.")


if __name__ == "__main__":

    test_fresh_database()
    test_legacy_database_migration()

    print("\nALL DATABASE TESTS PASSED.")

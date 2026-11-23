
from pathlib import Path

import numpy as np

from src.database import get_connection, DATABASE_PATH


# ==========================================
# PERSON MANAGEMENT
# ==========================================

def create_person(
    person_id: str,
    display_name: str | None = None,
    db_path: str | Path = DATABASE_PATH
) -> str:
    """Create a person record without overwriting an existing one."""

    if not person_id or not person_id.strip():
        raise ValueError("Person ID cannot be empty.")

    connection = get_connection(db_path)

    try:
        with connection:

            existing = connection.execute(
                """
                SELECT person_id
                FROM people
                WHERE person_id = ?
                """,
                (person_id,)
            ).fetchone()

            if existing:
                return person_id

            connection.execute(
                """
                INSERT INTO people (
                    person_id,
                    display_name
                )
                VALUES (?, ?)
                """,
                (person_id, display_name)
            )

        return person_id

    finally:
        connection.close()


# ==========================================
# FACE RECORD MANAGEMENT
# ==========================================

def save_face_record(
    image_path: str | Path,
    face_number: int,
    person_id: str | None = None,
    embedding=None,
    db_path: str | Path = DATABASE_PATH
) -> int:
    """
    Save a detected face and its embedding.

    Existing face records are preserved.
    Conflicting person assignments are rejected.
    """

    if face_number < 1:
        raise ValueError("Face number must be at least 1.")

    image_path = str(Path(image_path).resolve())

    embedding_blob = None

    if embedding is not None:

        embedding_array = np.asarray(
            embedding,
            dtype=np.float32
        ).flatten()

        if embedding_array.size == 0:
            raise ValueError("Embedding cannot be empty.")

        embedding_blob = embedding_array.tobytes()

    connection = get_connection(db_path)

    try:
        with connection:

            existing = connection.execute(
                """
                SELECT face_id, person_id
                FROM face_records
                WHERE image_path = ?
                  AND face_number = ?
                """,
                (image_path, face_number)
            ).fetchone()

            if existing:

                existing_person = existing["person_id"]

                if (
                    person_id is not None
                    and existing_person is not None
                    and existing_person != person_id
                ):
                    raise ValueError(
                        "This face is already assigned to "
                        f"{existing_person}. Assignment to "
                        f"{person_id} was rejected."
                    )

                return existing["face_id"]

            connection.execute(
                """
                INSERT INTO face_records (
                    image_path,
                    face_number,
                    person_id,
                    embedding
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    image_path,
                    face_number,
                    person_id,
                    embedding_blob
                )
            )

            result = connection.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()

            return result[0]

    finally:
        connection.close()


# ==========================================
# FACE RECORD LOOKUP
# ==========================================

def get_face_record_id(
    image_path: str | Path,
    face_number: int,
    db_path: str | Path = DATABASE_PATH
) -> int | None:
    """Retrieve the database ID for a detected face."""

    image_path = str(Path(image_path).resolve())

    connection = get_connection(db_path)

    try:

        row = connection.execute(
            """
            SELECT face_id
            FROM face_records
            WHERE image_path = ?
              AND face_number = ?
            """,
            (image_path, face_number)
        ).fetchone()

        return row["face_id"] if row else None

    finally:
        connection.close()


# ==========================================
# IDENTITY DECISION MANAGEMENT
# ==========================================

def save_identity_decision(
    candidate_type: str,
    person_a_id: str | None,
    decision: str,
    person_b_id: str | None = None,
    face_record_id: int | None = None,
    face_record_id_b: int | None = None,
    maximum_similarity: float | None = None,
    average_similarity: float | None = None,
    minimum_similarity: float | None = None,
    db_path: str | Path = DATABASE_PATH
) -> int:
    """
    Append a human identity review decision.

    Supported types:
    - GROUP_GROUP
    - UNMATCHED_FACE_GROUP
    - UNMATCHED_FACE_PAIR

    This function never merges people or assigns faces.
    """

    allowed_types = {
        "GROUP_GROUP",
        "UNMATCHED_FACE_GROUP",
        "UNMATCHED_FACE_PAIR"
    }

    allowed_decisions = {
        "SAME_PERSON",
        "DIFFERENT_PEOPLE",
        "SKIPPED"
    }

    if candidate_type not in allowed_types:
        raise ValueError(
            f"Invalid candidate type: {candidate_type}"
        )

    if decision not in allowed_decisions:
        raise ValueError("Invalid identity decision.")

    # --------------------------------------
    # GROUP vs GROUP
    # --------------------------------------

    if candidate_type == "GROUP_GROUP":

        if not person_a_id or not person_b_id:
            raise ValueError(
                "Both person IDs are required for group comparisons."
            )

        if person_a_id == person_b_id:
            raise ValueError(
                "A person cannot be compared against themselves."
            )

        if face_record_id is not None or face_record_id_b is not None:
            raise ValueError(
                "Group comparisons cannot contain face record IDs."
            )

    # --------------------------------------
    # UNMATCHED FACE vs GROUP
    # --------------------------------------

    elif candidate_type == "UNMATCHED_FACE_GROUP":

        if not person_a_id:
            raise ValueError("Person ID is required.")

        if person_b_id is not None:
            raise ValueError(
                "Person B must be empty for unmatched-face comparisons."
            )

        if face_record_id is None:
            raise ValueError(
                "An unmatched face record ID is required."
            )

        if face_record_id_b is not None:
            raise ValueError(
                "A second face record ID is not allowed for this type."
            )

    # --------------------------------------
    # UNMATCHED FACE vs UNMATCHED FACE
    # --------------------------------------

    elif candidate_type == "UNMATCHED_FACE_PAIR":

        if person_a_id is not None or person_b_id is not None:
            raise ValueError(
                "Unmatched face pairs cannot contain person IDs."
            )

        if face_record_id is None or face_record_id_b is None:
            raise ValueError(
                "Both face record IDs are required."
            )

        if face_record_id == face_record_id_b:
            raise ValueError(
                "A face cannot be compared against itself."
            )

        # Canonical ordering ensures the same pair has one identity.
        face_record_id, face_record_id_b = sorted(
            (face_record_id, face_record_id_b)
        )

    connection = get_connection(db_path)

    try:
        with connection:

            connection.execute(
                """
                INSERT INTO identity_decisions (
                    candidate_type,
                    person_a_id,
                    person_b_id,
                    face_record_id,
                    face_record_id_b,
                    decision,
                    maximum_similarity,
                    average_similarity,
                    minimum_similarity
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_type,
                    person_a_id,
                    person_b_id,
                    face_record_id,
                    face_record_id_b,
                    decision,
                    maximum_similarity,
                    average_similarity,
                    minimum_similarity
                )
            )

            result = connection.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()

            return result[0]

    finally:
        connection.close()


# ==========================================
# IDENTITY DECISION RETRIEVAL
# ==========================================

def get_identity_decisions(
    person_id: str | None = None,
    db_path: str | Path = DATABASE_PATH
) -> list[dict]:
    """
    Retrieve identity decisions.

    If person_id is supplied, retrieve decisions involving
    that person. Face-pair decisions have no person IDs and
    are returned when person_id is not supplied.
    """

    connection = get_connection(db_path)

    try:

        if person_id is None:

            rows = connection.execute(
                """
                SELECT *
                FROM identity_decisions
                ORDER BY decided_at DESC, decision_id DESC
                """
            ).fetchall()

        else:

            rows = connection.execute(
                """
                SELECT *
                FROM identity_decisions
                WHERE person_a_id = ?
                   OR person_b_id = ?
                ORDER BY decided_at DESC, decision_id DESC
                """,
                (person_id, person_id)
            ).fetchall()

        return [
            dict(row) for row in rows
        ]

    finally:
        connection.close()


# ==========================================
# PERSON FACE RETRIEVAL
# ==========================================

def get_person_faces(
    person_id: str,
    db_path: str | Path = DATABASE_PATH
) -> list[dict]:
    """
    Retrieve all face records assigned to a person.

    Embeddings are returned as NumPy float32 arrays.
    """

    connection = get_connection(db_path)

    try:

        rows = connection.execute(
            """
            SELECT *
            FROM face_records
            WHERE person_id = ?
            ORDER BY face_id
            """,
            (person_id,)
        ).fetchall()

        results = []

        for row in rows:

            face = dict(row)

            if face["embedding"] is not None:

                face["embedding"] = np.frombuffer(
                    face["embedding"],
                    dtype=np.float32
                ).copy()

            results.append(face)

        return results

    finally:
        connection.close()

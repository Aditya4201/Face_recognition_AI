
from itertools import combinations

from src.face_recognizer import compare_faces
from config.settings import REVIEW_MATCH_THRESHOLD


def _build_candidate(
    candidate_type: str,
    comparisons: list[dict],
    **metadata
) -> dict | None:
    """
    Calculate similarity statistics and prepare a candidate
    for human identity review.

    This function never merges or assigns identities.
    """

    if not comparisons:
        return None

    similarities = [
        item["similarity"] for item in comparisons
    ]

    maximum = max(similarities)

    # Only suggest candidates above the configured review threshold.
    if maximum < REVIEW_MATCH_THRESHOLD:
        return None

    sorted_comparisons = sorted(
        comparisons,
        key=lambda item: item["similarity"],
        reverse=True
    )

    return {
        "candidate_type": candidate_type,
        "maximum_similarity": maximum,
        "average_similarity": sum(similarities) / len(similarities),
        "minimum_similarity": min(similarities),
        "comparisons": sorted_comparisons,
        **metadata
    }


def find_duplicate_group_candidates(
    person_groups: list[dict],
    unmatched_faces: list[dict] | None = None
) -> list[dict]:
    """
    Find possible identity matches in three categories:

    1. Existing person group vs another existing person group.
    2. An unmatched face vs an existing person group.
    3. An unmatched face vs another unmatched face.

    Faces from the same photograph are never compared.

    This function only suggests candidates.
    It never merges or assigns identities.
    """

    if unmatched_faces is None:
        unmatched_faces = []

    candidates = []

    # ==================================================
    # 1. PERSON GROUP vs PERSON GROUP
    # ==================================================

    for i in range(len(person_groups)):
        for j in range(i + 1, len(person_groups)):

            group_a = person_groups[i]
            group_b = person_groups[j]

            comparisons = []

            for face_a in group_a["faces"]:
                for face_b in group_b["faces"]:

                    # Do not compare faces from the same photograph.
                    if face_a["image_path"] == face_b["image_path"]:
                        continue

                    similarity = compare_faces(
                        face_a["embedding"],
                        face_b["embedding"]
                    )

                    comparisons.append({
                        "face_a": face_a,
                        "face_b": face_b,
                        "similarity": similarity
                    })

            candidate = _build_candidate(
                candidate_type="GROUP_GROUP",
                comparisons=comparisons,
                person_a=group_a["person_id"],
                person_b=group_b["person_id"]
            )

            if candidate is not None:
                candidates.append(candidate)

    # ==================================================
    # 2. UNMATCHED FACE vs PERSON GROUP
    # ==================================================

    for unmatched_face in unmatched_faces:
        for group in person_groups:

            comparisons = []

            for group_face in group["faces"]:

                # Do not compare faces from the same photograph.
                if (
                    unmatched_face["image_path"]
                    == group_face["image_path"]
                ):
                    continue

                similarity = compare_faces(
                    unmatched_face["embedding"],
                    group_face["embedding"]
                )

                comparisons.append({
                    "face_a": group_face,
                    "face_b": unmatched_face,
                    "similarity": similarity
                })

            candidate = _build_candidate(
                candidate_type="UNMATCHED_FACE_GROUP",
                comparisons=comparisons,
                person_id=group["person_id"],
                unmatched_face=unmatched_face
            )

            if candidate is not None:
                candidates.append(candidate)

    # ==================================================
    # 3. UNMATCHED FACE vs UNMATCHED FACE
    # ==================================================

    for face_a, face_b in combinations(unmatched_faces, 2):

        # Faces from the same photograph cannot represent
        # two separate appearances of the same person.
        if face_a["image_path"] == face_b["image_path"]:
            continue

        similarity = compare_faces(
            face_a["embedding"],
            face_b["embedding"]
        )

        comparison = {
            "face_a": face_a,
            "face_b": face_b,
            "similarity": similarity
        }

        candidate = _build_candidate(
            candidate_type="UNMATCHED_FACE_PAIR",
            comparisons=[comparison],
            face_a=face_a,
            face_b=face_b
        )

        if candidate is not None:
            candidates.append(candidate)

    # ==================================================
    # 4. RANK ALL CANDIDATES
    # ==================================================

    candidates.sort(
        key=lambda item: (
            item["maximum_similarity"],
            item["average_similarity"]
        ),
        reverse=True
    )

    return candidates

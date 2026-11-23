
from pathlib import Path

from src.face_recognizer import compare_faces


def diagnose_unmatched_faces(
    faces: list[dict],
    clustering_results: dict
) -> list[dict]:

    person_groups = clustering_results["person_groups"]
    unmatched_faces = clustering_results["unmatched_faces"]
    threshold = clustering_results["similarity_threshold"]

    diagnostics = []

    for unmatched_face in unmatched_faces:

        candidates = []

        for group in person_groups:

            group_faces = group["faces"]

            # Check whether this photograph already appears
            # in the candidate person group.
            same_image_conflict = any(
                unmatched_face["image_path"] == member["image_path"]
                for member in group_faces
            )

            similarities = []

            for member in group_faces:

                # Same-image faces are not valid identity
                # comparison candidates for this grouping rule.
                if unmatched_face["image_path"] == member["image_path"]:
                    continue

                similarity = compare_faces(
                    unmatched_face["embedding"],
                    member["embedding"]
                )

                similarities.append(similarity)

            if not similarities:
                continue

            maximum = max(similarities)
            average = sum(similarities) / len(similarities)
            minimum = min(similarities)

            all_pairs_match = all(
                score >= threshold for score in similarities
            )

            if same_image_conflict:
                status = "BLOCKED_SAME_PHOTO"

            elif all_pairs_match:
                status = "ELIGIBLE_TO_MERGE"

            elif maximum >= threshold:
                status = "PARTIAL_MATCH"

            else:
                status = "BELOW_THRESHOLD"

            candidates.append({
                "person_id": group["person_id"],
                "maximum_similarity": maximum,
                "average_similarity": average,
                "minimum_similarity": minimum,
                "same_image_conflict": same_image_conflict,
                "all_pairs_match": all_pairs_match,
                "status": status
            })

        # Rank candidate groups by their strongest similarity.
        candidates.sort(
            key=lambda item: (
                item["maximum_similarity"],
                item["average_similarity"]
            ),
            reverse=True
        )

        diagnostics.append({
            "image_path": unmatched_face["image_path"],
            "face_number": unmatched_face["face_number"],
            "candidates": candidates
        })

    return diagnostics


def print_diagnostics(diagnostics: list[dict]) -> None:

    print("\nUNMATCHED FACE DIAGNOSTICS")
    print("=" * 65)

    for item in diagnostics:

        filename = Path(item["image_path"]).name

        print(f"\n{filename} | Face {item['face_number']}")
        print("-" * 65)

        if not item["candidates"]:
            print("No eligible comparison groups found.")
            continue

        # Show the strongest candidate group.
        best = item["candidates"][0]

        print(f"Closest group: {best['person_id']}")
        print(
            f"Maximum similarity: "
            f"{best['maximum_similarity']:.4f}"
        )
        print(
            f"Average similarity: "
            f"{best['average_similarity']:.4f}"
        )
        print(
            f"Minimum similarity: "
            f"{best['minimum_similarity']:.4f}"
        )
        print(f"Status: {best['status']}")

        # Show other candidate groups if available.
        for candidate in item["candidates"][1:]:

            print(
                f"Other candidate: {candidate['person_id']} | "
                f"Max: {candidate['maximum_similarity']:.4f} | "
                f"Avg: {candidate['average_similarity']:.4f} | "
                f"Min: {candidate['minimum_similarity']:.4f} | "
                f"Status: {candidate['status']}"
            )

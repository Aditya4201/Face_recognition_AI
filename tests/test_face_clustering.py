
from pathlib import Path

from config.settings import (
    FACE_DETECTION_THRESHOLD,
    REVIEW_MATCH_THRESHOLD,
)

from src.image_scanner import scan_images
from src.face_recognizer import (
    extract_face_embeddings,
    compare_faces,
)
from src.face_clustering import cluster_faces
from src.identity_review import find_duplicate_group_candidates


TARGET_PHOTOS = {
    "20260523_185835",
    "20260523_185823",
    "20260523_181748",
}


def face_label(face):
    filename = Path(face["image_path"]).name
    return f"{filename} | Face {face['face_number']}"


def main():

    print("=" * 70)
    print("              FACESORT IDENTITY DIAGNOSTIC")
    print("=" * 70)

    photo_folder = input("\nEnter your photo folder path: ").strip()

    detector_model = Path(
        "models/face_detection_yunet_2023mar.onnx"
    )

    recognizer_model = Path(
        "models/face_recognition_sface_2021dec.onnx"
    )

    # --------------------------------------------------
    # STEP 1: SCAN PHOTOGRAPHS
    # --------------------------------------------------

    print("\nSTEP 1: SCANNING PHOTOGRAPHS")
    print("-" * 70)

    images = scan_images(photo_folder)

    print(f"Images discovered: {len(images)}")

    # --------------------------------------------------
    # STEP 2: EXTRACT FACE EMBEDDINGS
    # --------------------------------------------------

    print("\nSTEP 2: EXTRACTING FACE EMBEDDINGS")
    print("-" * 70)

    faces = extract_face_embeddings(
        images,
        detector_model,
        recognizer_model,
        score_threshold=FACE_DETECTION_THRESHOLD,
    )

    print(f"Faces detected and embedded: {len(faces)}")

    # --------------------------------------------------
    # STEP 3: AUTOMATIC CLUSTERING
    # --------------------------------------------------

    print("\nSTEP 3: AUTOMATIC CLUSTERING")
    print("-" * 70)

    results = cluster_faces(faces)

    person_groups = results["person_groups"]
    unmatched_faces = results["unmatched_faces"]

    print(f"Clustering threshold: {results['similarity_threshold']}")
    print(f"Person groups: {len(person_groups)}")
    print(f"Unmatched faces: {len(unmatched_faces)}")

    # Map each face to its automatically assigned group.
    assignments = {}

    for group in person_groups:
        for face in group["faces"]:
            key = (
                str(face["image_path"]),
                face["face_number"],
            )
            assignments[key] = group["person_id"]

    # --------------------------------------------------
    # STEP 4: DISPLAY ALL PERSON GROUPS
    # --------------------------------------------------

    print("\nSTEP 4: AUTOMATICALLY IDENTIFIED PEOPLE")
    print("=" * 70)

    for group in person_groups:

        print(f"\n{group['person_id']}")
        print("-" * 50)

        for face in group["faces"]:
            print(f"  {face_label(face)}")

        print(f"Total appearances: {len(group['faces'])}")

    # --------------------------------------------------
    # STEP 5: DISPLAY ALL FACES IN TARGET PHOTOGRAPHS
    # --------------------------------------------------

    print("\nSTEP 5: TARGET PHOTOGRAPH INSPECTION")
    print("=" * 70)

    for face in faces:

        filename = Path(face["image_path"]).stem

        if not any(
            filename.startswith(target)
            for target in TARGET_PHOTOS
        ):
            continue

        key = (
            str(face["image_path"]),
            face["face_number"],
        )

        assigned_person = assignments.get(key, "UNMATCHED")

        print(
            f"{face_label(face)}"
            f" | Assignment: {assigned_person}"
        )

    # --------------------------------------------------
    # STEP 6: GENERATE NORMAL IDENTITY REVIEW CANDIDATES
    # --------------------------------------------------

    print("\nSTEP 6: IDENTITY REVIEW CANDIDATES")
    print("=" * 70)

    review_candidates = find_duplicate_group_candidates(
        person_groups,
        unmatched_faces,
    )

    print(f"Review candidates generated: {len(review_candidates)}")

    for index, candidate in enumerate(review_candidates, start=1):

        print(f"\nCandidate {index}")
        print("-" * 50)

        print(f"Type: {candidate['candidate_type']}")

        if candidate["candidate_type"] == "GROUP_GROUP":
            print(f"Person A: {candidate['person_a']}")
            print(f"Person B: {candidate['person_b']}")

        elif candidate["candidate_type"] == "UNMATCHED_FACE_GROUP":
            print(f"Person: {candidate['person_id']}")
            print(
                f"Unmatched face: "
                f"{face_label(candidate['unmatched_face'])}"
            )

        print(
            f"Maximum similarity: "
            f"{candidate['maximum_similarity']:.4f}"
        )

        print(
            f"Average similarity: "
            f"{candidate['average_similarity']:.4f}"
        )

        print(
            f"Minimum similarity: "
            f"{candidate['minimum_similarity']:.4f}"
        )

    # --------------------------------------------------
    # STEP 7: DEEP DIAGNOSTIC
    # COMPARE EVERY UNMATCHED FACE AGAINST EVERY PERSON
    # --------------------------------------------------

    print("\nSTEP 7: ALL UNMATCHED-FACE COMPARISONS")
    print("=" * 70)

    print(
        "This section deliberately displays scores below "
        "the review threshold too."
    )

    print(f"Review threshold: {REVIEW_MATCH_THRESHOLD:.4f}")

    for unmatched_face in unmatched_faces:

        print(f"\nUNMATCHED: {face_label(unmatched_face)}")
        print("-" * 60)

        comparisons_found = False

        for group in person_groups:

            similarities = []

            for group_face in group["faces"]:

                if (
                    unmatched_face["image_path"]
                    == group_face["image_path"]
                ):
                    continue

                similarity = compare_faces(
                    unmatched_face["embedding"],
                    group_face["embedding"],
                )

                similarities.append(
                    (similarity, group_face)
                )

            if not similarities:
                continue

            comparisons_found = True

            scores = [
                item[0] for item in similarities
            ]

            maximum = max(scores)
            average = sum(scores) / len(scores)
            minimum = min(scores)

            status = (
                "ABOVE REVIEW THRESHOLD"
                if maximum >= REVIEW_MATCH_THRESHOLD
                else "BELOW REVIEW THRESHOLD"
            )

            print(f"\nCompared against: {group['person_id']}")
            print(f"Comparisons: {len(similarities)}")
            print(f"Maximum: {maximum:.4f}")
            print(f"Average: {average:.4f}")
            print(f"Minimum: {minimum:.4f}")
            print(f"Status: {status}")

            print("Top 3 matching faces:")

            top_matches = sorted(
                similarities,
                key=lambda item: item[0],
                reverse=True,
            )[:3]

            for similarity, matched_face in top_matches:
                print(
                    f"  {similarity:.4f} | "
                    f"{face_label(matched_face)}"
                )

        if not comparisons_found:
            print("No valid cross-photo comparisons available.")

    # --------------------------------------------------
    # STEP 8: UNMATCHED FACE SUMMARY
    # --------------------------------------------------

    print("\nSTEP 8: UNMATCHED FACES")
    print("=" * 70)

    for face in unmatched_faces:
        print(f"  {face_label(face)}")

    # --------------------------------------------------
    # STEP 9: FINAL SUMMARY
    # --------------------------------------------------

    grouped_count = sum(
        len(group["faces"])
        for group in person_groups
    )

    print("\nFINAL SUMMARY")
    print("=" * 70)

    print(f"Images scanned: {len(images)}")
    print(f"Faces processed: {len(faces)}")
    print(f"Person groups: {len(person_groups)}")
    print(f"Automatically grouped faces: {grouped_count}")
    print(f"Unmatched faces: {len(unmatched_faces)}")
    print(f"Review candidates: {len(review_candidates)}")

    print(
        f"Total accounted faces: "
        f"{grouped_count + len(unmatched_faces)}"
    )

    print("=" * 70)
    print("             DIAGNOSTIC COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

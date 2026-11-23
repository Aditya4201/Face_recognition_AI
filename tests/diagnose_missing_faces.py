
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    FACE_DETECTION_THRESHOLD,
    REVIEW_MATCH_THRESHOLD
)

from src.image_scanner import scan_images
from src.face_recognizer import (
    extract_face_embeddings,
    compare_faces
)
from src.face_clustering import cluster_faces


# ==================================================
# PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DETECTOR_MODEL = (
    BASE_DIR
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = (
    BASE_DIR
    / "models"
    / "face_recognition_sface_2021dec.onnx"
)

# Faces we want to investigate.
TARGETS = {
    "20260523_181748.jpg": 4,
    "20260523_185823.jpg": 4,
    "20260523_185835.jpg": 4,
}

# Output directory for aligned face images.
ALIGNED_PREVIEW_DIR = (
    BASE_DIR
    / "test"
    / "aligned_face_diagnostics"
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def get_person_id(face, person_groups):
    """
    Find the automatically assigned person group
    for a particular face.

    Returns UNMATCHED if the face is not grouped.
    """

    target_path = Path(face["image_path"]).resolve()

    for group in person_groups:

        for group_face in group["faces"]:

            group_path = Path(
                group_face["image_path"]
            ).resolve()

            if (
                target_path == group_path
                and face["face_number"]
                == group_face["face_number"]
            ):
                return group["person_id"]

    return "UNMATCHED"


def save_aligned_face(face, filename, face_number):
    """
    Save the SFace-aligned face image.

    Original photographs are never modified.
    """

    aligned_face = face.get("aligned_face")

    if aligned_face is None:
        print("WARNING: Aligned face image unavailable.")
        return None

    ALIGNED_PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_name = (
        f"{Path(filename).stem}"
        f"_Face_{face_number}.jpg"
    )

    output_path = (
        ALIGNED_PREVIEW_DIR / output_name
    )

    success = cv2.imwrite(
        str(output_path),
        aligned_face,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if success:
        print(f"Aligned preview saved: {output_path}")
        return output_path

    print(f"ERROR: Could not save {output_path}")
    return None


# ==================================================
# MAIN DIAGNOSTIC
# ==================================================

def main():

    print("\n" + "=" * 65)
    print("FACESORT AI - MISSING FACE DIAGNOSTIC")
    print("=" * 65)

    # ----------------------------------------------
    # Validate model files
    # ----------------------------------------------

    if not DETECTOR_MODEL.exists():
        print(f"ERROR: YuNet model not found: {DETECTOR_MODEL}")
        return

    if not RECOGNIZER_MODEL.exists():
        print(f"ERROR: SFace model not found: {RECOGNIZER_MODEL}")
        return

    # ----------------------------------------------
    # Get photograph folder
    # ----------------------------------------------

    folder = input(
        "\nEnter your original photograph folder path: "
    ).strip().strip('"')

    if not Path(folder).is_dir():
        print("ERROR: Invalid photograph folder.")
        return

    # ----------------------------------------------
    # Scan photographs
    # ----------------------------------------------

    print("\nScanning photographs...")

    image_paths = scan_images(
        folder,
        recursive=True
    )

    print(f"Images found: {len(image_paths)}")

    # ----------------------------------------------
    # Extract face embeddings
    # ----------------------------------------------

    print("\nExtracting face embeddings...")

    faces = extract_face_embeddings(
        image_paths,
        DETECTOR_MODEL,
        RECOGNIZER_MODEL,
        score_threshold=FACE_DETECTION_THRESHOLD
    )

    print(f"Total faces embedded: {len(faces)}")

    if not faces:
        print("ERROR: No face embeddings generated.")
        return

    # ----------------------------------------------
    # Generate person groups
    # ----------------------------------------------

    print("\nRunning complete-link clustering...")

    results = cluster_faces(faces)

    person_groups = results["person_groups"]
    unmatched_faces = results["unmatched_faces"]

    print(f"Person groups: {len(person_groups)}")
    print(f"Unmatched faces: {len(unmatched_faces)}")
    print(
        f"Clustering threshold: "
        f"{results['similarity_threshold']:.3f}"
    )

    # ----------------------------------------------
    # Locate target faces
    # ----------------------------------------------

    target_faces = []

    print("\n" + "=" * 65)
    print("TARGET FACE EMBEDDING DETAILS")
    print("=" * 65)

    for filename, face_number in TARGETS.items():

        matches = [
            face for face in faces
            if (
                Path(face["image_path"]).name.lower()
                == filename.lower()
                and face["face_number"] == face_number
            )
        ]

        if not matches:

            print(
                f"\nNOT FOUND: {filename}, "
                f"Face {face_number}"
            )

            continue

        target = matches[0]

        embedding = np.asarray(
            target["embedding"]
        )

        person_id = get_person_id(
            target,
            person_groups
        )

        print("\n" + "-" * 65)
        print(f"Photograph: {filename}")
        print(f"Face number: {face_number}")
        print(f"Person group: {person_id}")
        print(f"Embedding dimensions: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
        print(
            f"Embedding finite: "
            f"{np.isfinite(embedding).all()}"
        )
        print(
            f"Embedding values: "
            f"{len(embedding)}"
        )

        if not np.isfinite(embedding).all():
            print("WARNING: Embedding contains invalid values.")

        if np.linalg.norm(embedding) == 0:
            print("WARNING: Embedding is zero-valued.")

        # Save aligned face for visual inspection.
        save_aligned_face(
            target,
            filename,
            face_number
        )

        target_faces.append(target)

    if not target_faces:

        print("\nNo target faces were found.")
        return

    # ----------------------------------------------
    # Compare targets against every other face
    # ----------------------------------------------

    for target in target_faces:

        target_filename = Path(
            target["image_path"]
        ).name

        print("\n" + "#" * 65)
        print(
            f"SIMILARITY REPORT: "
            f"{target_filename} "
            f"| Face {target['face_number']}"
        )
        print("#" * 65)

        scores = []

        target_path = Path(
            target["image_path"]
        ).resolve()

        for other in faces:

            if other is target:
                continue

            other_path = Path(
                other["image_path"]
            ).resolve()

            # Never compare two faces from the same photograph.
            if target_path == other_path:
                continue

            similarity = compare_faces(
                target["embedding"],
                other["embedding"]
            )

            person_id = get_person_id(
                other,
                person_groups
            )

            scores.append({
                "similarity": similarity,
                "filename": Path(
                    other["image_path"]
                ).name,
                "face_number": other["face_number"],
                "person": person_id
            })

        # Rank every comparison from highest to lowest.
        scores.sort(
            key=lambda item: item["similarity"],
            reverse=True
        )

        print("\nTOP 10 MATCHES:")

        for item in scores[:10]:

            print(
                f"{item['similarity']:.4f} | "
                f"{item['person']:<12} | "
                f"{item['filename']} | "
                f"Face {item['face_number']}"
            )

        if scores:

            best = scores[0]

            print("\nBEST MATCH:")

            print(
                f"Similarity: {best['similarity']:.4f}"
            )

            print(
                f"Photograph: {best['filename']}"
            )

            print(
                f"Face number: {best['face_number']}"
            )

            print(
                f"Person group: {best['person']}"
            )

            print(
                "\nAbove review threshold:",
                best["similarity"] >= REVIEW_MATCH_THRESHOLD
            )

            print(
                "Above clustering threshold:",
                best["similarity"]
                >= results["similarity_threshold"]
            )

        else:
            print("No other faces available for comparison.")

    # ----------------------------------------------
    # Final summary
    # ----------------------------------------------

    print("\n" + "=" * 65)
    print("DIAGNOSTIC COMPLETED")
    print("=" * 65)

    print(f"Total faces embedded: {len(faces)}")
    print(f"Target faces found: {len(target_faces)}")

    print(
        "\nAligned face images are saved in:"
    )

    print(ALIGNED_PREVIEW_DIR)

    print("\nCurrent thresholds:")
    print(
        f"Detection: {FACE_DETECTION_THRESHOLD:.3f}"
    )
    print(
        f"Identity review: {REVIEW_MATCH_THRESHOLD:.3f}"
    )
    print(
        f"Clustering: "
        f"{results['similarity_threshold']:.3f}"
    )

    print("\nNo original photographs were modified.")
    print("No identity groups were changed.")


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()

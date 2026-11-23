
from pathlib import Path
import csv

import cv2
import numpy as np

from src.face_recognizer import compare_faces


def generate_verification_previews(
    faces: list[dict],
    output_dir: str | Path,
    max_previews: int = 50
) -> list[dict]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove previous generated comparison images.
    for file in output_dir.glob("comparison_*.jpg"):
        if file.is_file():
            file.unlink()

    comparisons = []

    # Compare every pair of faces from different photographs.
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):

            face_a = faces[i]
            face_b = faces[j]

            # Skip faces from the same photograph.
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

    # Sort all comparisons from highest to lowest similarity.
    comparisons.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    # Save every comparison in a CSV report.
    report_path = output_dir / "all_comparisons.csv"

    with report_path.open(
        mode="w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow([
            "Image A",
            "Face A",
            "Image B",
            "Face B",
            "Cosine Similarity"
        ])

        for match in comparisons:

            face_a = match["face_a"]
            face_b = match["face_b"]

            writer.writerow([
                Path(face_a["image_path"]).name,
                face_a["face_number"],
                Path(face_b["image_path"]).name,
                face_b["face_number"],
                round(match["similarity"], 4)
            ])

    # Generate previews only for the highest-scoring comparisons.
    selected = comparisons[:max_previews]

    for index, match in enumerate(selected, start=1):

        face_a = match["face_a"]
        face_b = match["face_b"]
        similarity = match["similarity"]

        size = 280

        # Resize aligned faces into square panels.
        image_a = cv2.resize(
            face_a["aligned_face"],
            (size, size)
        )

        image_b = cv2.resize(
            face_b["aligned_face"],
            (size, size)
        )

        # Create comparison canvas.
        canvas = np.full(
            (450, 620, 3),
            255,
            dtype=np.uint8
        )

        canvas[90:370, 20:300] = image_a
        canvas[90:370, 320:600] = image_b

        # Comparison ID.
        cv2.putText(
            canvas,
            f"Comparison #{index}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        # Similarity score.
        cv2.putText(
            canvas,
            f"Similarity: {similarity:.4f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 100, 0),
            2
        )

        name_a = Path(face_a["image_path"]).name
        name_b = Path(face_b["image_path"]).name

        # First photograph details.
        cv2.putText(
            canvas,
            f"A: {name_a[:32]} | Face {face_a['face_number']}",
            (20, 395),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1
        )

        # Second photograph details.
        cv2.putText(
            canvas,
            f"B: {name_b[:32]} | Face {face_b['face_number']}",
            (20, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            1
        )

        preview_path = output_dir / f"comparison_{index:03d}.jpg"

        if not cv2.imwrite(str(preview_path), canvas):
            raise OSError(
                f"Could not save comparison preview: {preview_path}"
            )

        match["preview_path"] = str(preview_path)

    print(f"\nTotal valid comparisons: {len(comparisons)}")
    print(f"Visual previews generated: {len(selected)}")
    print(f"Complete comparison report: {report_path.resolve()}")

    return selected

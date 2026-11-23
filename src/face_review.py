
from pathlib import Path

import cv2
import numpy as np


def _create_contact_sheet(
    faces: list[dict],
    title: str,
    output_path: Path,
    columns: int = 4
) -> None:

    if not faces:
        return

    # Layout configuration.
    thumbnail_size = 180
    cell_width = 220
    cell_height = 240
    header_height = 65
    padding = 15

    rows = (len(faces) + columns - 1) // columns

    canvas_width = columns * cell_width + padding
    canvas_height = header_height + rows * cell_height + padding

    canvas = np.full(
        (canvas_height, canvas_width, 3),
        255,
        dtype=np.uint8
    )

    # Add sheet title.
    cv2.putText(
        canvas,
        title,
        (padding, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2
    )

    for index, face in enumerate(faces):

        row = index // columns
        column = index % columns

        x = padding + column * cell_width
        y = header_height + row * cell_height

        aligned_face = face.get("aligned_face")

        if aligned_face is None:
            continue

        # Resize face to a consistent square thumbnail.
        thumbnail = cv2.resize(
            aligned_face,
            (thumbnail_size, thumbnail_size)
        )

        canvas[
            y:y + thumbnail_size,
            x:x + thumbnail_size
        ] = thumbnail

        # Add a border around the thumbnail.
        cv2.rectangle(
            canvas,
            (x, y),
            (x + thumbnail_size, y + thumbnail_size),
            (0, 0, 0),
            1
        )

        # Source photograph details.
        filename = Path(face["image_path"]).name
        face_number = face["face_number"]

        # Split long filenames across two lines.
        line_1 = filename[:28]
        line_2 = filename[28:56]

        cv2.putText(
            canvas,
            line_1,
            (x, y + thumbnail_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1
        )

        if line_2:
            cv2.putText(
                canvas,
                line_2,
                (x, y + thumbnail_size + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 0),
                1
            )

        cv2.putText(
            canvas,
            f"Face {face_number}",
            (x, y + thumbnail_size + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 100, 0),
            1
        )

    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not save contact sheet: {output_path}")


def generate_cluster_reviews(
    clustering_results: dict,
    output_dir: str | Path,
    unmatched_per_sheet: int = 12,
    columns: int = 4
) -> list[Path]:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove only previously generated review sheets.
    for pattern in (
        "Person_*.jpg",
        "Unmatched_*.jpg"
    ):
        for file in output_dir.glob(pattern):
            if file.is_file():
                file.unlink()

    generated_files = []

    person_groups = clustering_results["person_groups"]
    unmatched_faces = clustering_results["unmatched_faces"]

    # Generate one contact sheet for each person group.
    for group in person_groups:

        person_id = group["person_id"]
        group_faces = group["faces"]

        output_path = output_dir / f"{person_id}.jpg"

        _create_contact_sheet(
            faces=group_faces,
            title=f"{person_id} | {len(group_faces)} appearances",
            output_path=output_path,
            columns=columns
        )

        generated_files.append(output_path)

    # Generate multiple sheets for unmatched faces.
    for start in range(0, len(unmatched_faces), unmatched_per_sheet):

        page_faces = unmatched_faces[
            start:start + unmatched_per_sheet
        ]

        page_number = (start // unmatched_per_sheet) + 1

        output_path = output_dir / (
            f"Unmatched_{page_number:03d}.jpg"
        )

        _create_contact_sheet(
            faces=page_faces,
            title=(
                f"Unmatched Faces | "
                f"Page {page_number} | "
                f"{len(page_faces)} faces"
            ),
            output_path=output_path,
            columns=columns
        )

        generated_files.append(output_path)

    # Generate a text summary.
    summary_path = output_dir / "review_summary.txt"

    with summary_path.open(
        "w",
        encoding="utf-8"
    ) as summary:

        summary.write("FACE CLUSTER REVIEW SUMMARY\n")
        summary.write("=" * 45 + "\n\n")

        summary.write(
            f"Similarity threshold: "
            f"{clustering_results['similarity_threshold']}\n"
        )

        summary.write(
            f"Person groups: {len(person_groups)}\n"
        )

        summary.write(
            f"Unmatched faces: {len(unmatched_faces)}\n\n"
        )

        for group in person_groups:

            summary.write(
                f"\n{group['person_id']}\n"
            )

            summary.write("-" * 35 + "\n")

            for face in group["faces"]:

                summary.write(
                    f"{Path(face['image_path']).name} | "
                    f"Face {face['face_number']}\n"
                )

        summary.write("\n\nUNMATCHED FACES\n")
        summary.write("=" * 45 + "\n")

        for face in unmatched_faces:

            summary.write(
                f"{Path(face['image_path']).name} | "
                f"Face {face['face_number']}\n"
            )

    generated_files.append(summary_path)

    return generated_files

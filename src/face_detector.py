
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


def count_faces(
    image_paths: list[Path],
    model_path: str | Path,
    preview_dir: str | Path,
    score_threshold: float = 0.6
) -> dict:
    """
    Detect faces using YuNet and save preview copies.

    All previews are saved as JPEG, regardless of the
    original photograph format.

    Original photographs are never modified.
    """

    model_path = Path(model_path)
    preview_dir = Path(preview_dir)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Face detection model not found: {model_path}"
        )

    # Create preview directory.
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Remove previous generated previews.
    supported_formats = {
        ".jpg", ".jpeg", ".png", ".webp",
        ".bmp", ".tiff", ".tif", ".heic", ".heif"
    }

    for file in preview_dir.iterdir():
        if file.is_file() and file.suffix.lower() in supported_formats:
            file.unlink()

    # Initialize YuNet.
    detector = cv2.FaceDetectorYN.create(
        model=str(model_path),
        config="",
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=0.3,
        top_k=5000
    )

    total_faces = 0
    images_with_faces = 0
    unreadable_images = 0
    previews_saved = 0
    preview_save_failures = 0

    for index, image_path in enumerate(image_paths, start=1):

        image_path = Path(image_path)

        # Read HEIC/HEIF using Pillow.
        if image_path.suffix.lower() in {".heic", ".heif"}:

            pil_image = Image.open(image_path).convert("RGB")

            image = cv2.cvtColor(
                np.array(pil_image),
                cv2.COLOR_RGB2BGR
            )

        else:
            image = cv2.imread(str(image_path))

        if image is None:
            unreadable_images += 1
            continue

        height, width = image.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is not None and len(faces) > 0:

            total_faces += len(faces)
            images_with_faces += 1

            # Draw a green rectangle around every detected face.
            for face_number, face in enumerate(faces, start=1):

                x, y, w, h = face[:4].astype(int)

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    f"Face {face_number}",
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Always save previews as JPEG.
        # This avoids OpenCV writer errors for HEIC/HEIF.
        preview_name = (
            f"{index:04d}_{image_path.stem}_preview.jpg"
        )

        preview_path = preview_dir / preview_name

        save_success = cv2.imwrite(
            str(preview_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95]
        )

        if save_success:
            previews_saved += 1
        else:
            preview_save_failures += 1

    return {
        "total_images": len(image_paths),
        "images_processed": len(image_paths) - unreadable_images,
        "images_with_faces": images_with_faces,
        "total_faces": total_faces,
        "unreadable_images": unreadable_images,
        "previews_saved": previews_saved,
        "preview_save_failures": preview_save_failures
    }

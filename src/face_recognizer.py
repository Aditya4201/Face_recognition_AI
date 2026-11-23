
from pathlib import Path

import cv2
import numpy as np


def extract_face_embeddings(
    image_paths: list[Path],
    detector_model_path: str | Path,
    recognizer_model_path: str | Path,
    score_threshold: float = 0.6
) -> list[dict]:

    detector_model_path = Path(detector_model_path)
    recognizer_model_path = Path(recognizer_model_path)

    # Validate model files.
    if not detector_model_path.exists():
        raise FileNotFoundError(
            f"YuNet model not found: {detector_model_path}"
        )

    if not recognizer_model_path.exists():
        raise FileNotFoundError(
            f"SFace model not found: {recognizer_model_path}"
        )

    # Initialize YuNet face detector.
    detector = cv2.FaceDetectorYN.create(
        model=str(detector_model_path),
        config="",
        input_size=(320, 320),
        score_threshold=score_threshold,
        nms_threshold=0.3,
        top_k=5000
    )

    # Initialize SFace face recognizer.
    recognizer = cv2.FaceRecognizerSF.create(
        model=str(recognizer_model_path),
        config=""
    )

    results = []

    for image_path in image_paths:

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        if faces is None:
            continue

        for face_number, face in enumerate(faces, start=1):

            # Align and crop the detected face.
            aligned_face = recognizer.alignCrop(image, face)

            # Generate the face embedding.
            embedding = recognizer.feature(aligned_face)

            results.append({
    "image_path": str(image_path),
    "face_number": face_number,
    "embedding": embedding.flatten(),
    "aligned_face": aligned_face.copy()
})

    return results


def compare_faces(
    embedding_1: np.ndarray,
    embedding_2: np.ndarray
) -> float:

    # Calculate cosine similarity.
    embedding_1 = np.asarray(embedding_1).flatten()
    embedding_2 = np.asarray(embedding_2).flatten()

    norm_1 = np.linalg.norm(embedding_1)
    norm_2 = np.linalg.norm(embedding_2)

    if norm_1 == 0 or norm_2 == 0:
        raise ValueError("Cannot compare zero-valued embeddings.")

    similarity = np.dot(
        embedding_1,
        embedding_2
    ) / (norm_1 * norm_2)

    return float(similarity)


from pathlib import Path

from src.image_scanner import scan_images
from src.face_recognizer import extract_face_embeddings
from src.face_verifier import generate_verification_previews


def main():

    photo_folder = input("Enter your photo folder path: ")

    detector_model = Path(
        "models/face_detection_yunet_2023mar.onnx"
    )

    recognizer_model = Path(
        "models/face_recognition_sface_2021dec.onnx"
    )

    preview_folder = Path("test/face_verifications")

    # Step 1: Scan photographs.
    images = scan_images(photo_folder)

    print(f"\nImages discovered: {len(images)}")

    # Step 2: Generate face embeddings.
    print("\nGenerating face embeddings...")

    faces = extract_face_embeddings(
        images,
        detector_model,
        recognizer_model,
        score_threshold=0.6
    )

    print(f"Faces processed: {len(faces)}")

    # Step 3: Generate comparison previews.
    print("\nGenerating visual comparisons...")

    comparisons = generate_verification_previews(
    faces,
    preview_folder,
    max_previews=50
    )

    print("\nVerification previews generated!")
    print("-" * 45)

    for match in comparisons:

        face_a = match["face_a"]
        face_b = match["face_b"]

        print(
            f"{Path(face_a['image_path']).name} "
            f"(Face {face_a['face_number']})"
        )

        print(
            f"{Path(face_b['image_path']).name} "
            f"(Face {face_b['face_number']})"
        )

        print(f"Similarity: {match['similarity']:.4f}")
        print(f"Preview: {match['preview_path']}")
        print("-" * 45)

    print(f"\nOpen this folder: {preview_folder.resolve()}")


if __name__ == "__main__":
    main()

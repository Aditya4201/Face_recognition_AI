
from pathlib import Path

from src.image_scanner import scan_images
from src.face_recognizer import (
    extract_face_embeddings,
    compare_faces
)


def main():

    photo_folder = input("Enter your photo folder path: ")

    detector_model = Path(
        "models/face_detection_yunet_2023mar.onnx"
    )

    recognizer_model = Path(
        "models/face_recognition_sface_2021dec.onnx"
    )

    images = scan_images(photo_folder)

    print(f"\nImages discovered: {len(images)}")
    print("Generating face embeddings...")

    faces = extract_face_embeddings(
        images,
        detector_model,
        recognizer_model,
        score_threshold=0.6
    )

    print("\nEmbedding generation completed!")
    print(f"Total faces recognised: {len(faces)}")

    for face in faces:
        print(
            f"{Path(face['image_path']).name} | "
            f"Face {face['face_number']} | "
            f"Embedding dimensions: "
            f"{len(face['embedding'])}"
        )

    
    # Compare faces across different photographs.
    comparisons = []

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):

            # Skip comparisons within the same photograph.
            if faces[i]["image_path"] == faces[j]["image_path"]:
                continue

            similarity = compare_faces(
                faces[i]["embedding"],
                faces[j]["embedding"]
            )

            comparisons.append({
                "image_1": Path(faces[i]["image_path"]).name,
                "face_1": faces[i]["face_number"],
                "image_2": Path(faces[j]["image_path"]).name,
                "face_2": faces[j]["face_number"],
                "similarity": similarity
            })

    # Sort from highest to lowest similarity.
    comparisons.sort(
        key=lambda item: item["similarity"],
        reverse=True
    )

    print("\nTop 10 potential matches across photographs:")
    print("-" * 75)

    for match in comparisons[:10]:
        print(
            f"{match['image_1']} (Face {match['face_1']})"
            f" <-> "
            f"{match['image_2']} (Face {match['face_2']})"
        )
        print(f"Similarity: {match['similarity']:.4f}")
        print("-" * 75)

if __name__ == "__main__":
    main()

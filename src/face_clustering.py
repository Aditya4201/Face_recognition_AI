
from src.face_recognizer import compare_faces


SIMILARITY_THRESHOLD = 0.65


def cluster_faces(faces: list[dict]) -> dict:
    """
    Group faces using conservative complete-link clustering.

    Every face starts in its own cluster.
    Two clusters are merged only when every cross-cluster
    similarity meets the configured threshold.

    Faces from the same photograph cannot belong to the
    same person group.
    """

    if not faces:
        return {
            "person_groups": [],
            "unmatched_faces": [],
            "similarity_threshold": SIMILARITY_THRESHOLD
        }

    # Initially, every face belongs to its own cluster.
    clusters = [[index] for index in range(len(faces))]

    # Calculate pairwise similarities once.
    similarities = {}

    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):

            # Do not compare faces from the same photograph.
            if faces[i]["image_path"] == faces[j]["image_path"]:
                continue

            similarity = compare_faces(
                faces[i]["embedding"],
                faces[j]["embedding"]
            )

            similarities[(i, j)] = similarity

    def get_similarity(i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        return similarities.get(key, -1.0)

    # Process strongest matches first.
    candidate_matches = sorted(
        similarities.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Merge clusters using complete-link clustering.
    for (face_i, face_j), similarity in candidate_matches:

        if similarity < SIMILARITY_THRESHOLD:
            break

        cluster_a = next(
            (cluster for cluster in clusters if face_i in cluster),
            None
        )

        cluster_b = next(
            (cluster for cluster in clusters if face_j in cluster),
            None
        )

        if cluster_a is None or cluster_b is None:
            continue

        if cluster_a is cluster_b:
            continue

        # Prevent multiple faces from the same photograph
        # from entering one person group.
        image_paths_a = {
            faces[index]["image_path"]
            for index in cluster_a
        }

        image_paths_b = {
            faces[index]["image_path"]
            for index in cluster_b
        }

        if image_paths_a.intersection(image_paths_b):
            continue

        # Every cross-cluster comparison must pass.
        all_pairs_match = all(
            get_similarity(a, b) >= SIMILARITY_THRESHOLD
            for a in cluster_a
            for b in cluster_b
        )

        if not all_pairs_match:
            continue

        cluster_a.extend(cluster_b)
        clusters.remove(cluster_b)

    # Separate person groups and unmatched faces.
    person_groups = []
    unmatched_faces = []

    for cluster in clusters:

        if len(cluster) >= 2:
            person_groups.append(cluster)
        else:
            unmatched_faces.extend(
                faces[index] for index in cluster
            )

    # Sort groups for consistent person IDs.
    person_groups.sort(
        key=lambda group: min(
            faces[index]["image_path"] for index in group
        )
    )

    formatted_groups = [
        {
            "person_id": f"Person_{index:03d}",
            "faces": [
                faces[face_index]
                for face_index in group
            ]
        }
        for index, group in enumerate(person_groups, start=1)
    ]

    return {
        "person_groups": formatted_groups,
        "unmatched_faces": unmatched_faces,
        "similarity_threshold": SIMILARITY_THRESHOLD
    }

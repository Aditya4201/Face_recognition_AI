
import json
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

# Enable HEIC and HEIF image support.
register_heif_opener()

from config.settings import (
    FACE_DETECTION_THRESHOLD,
    REVIEW_MATCH_THRESHOLD
)

from src.database import (
    initialize_database,
    DATABASE_PATH
)

from src.identity_repository import (
    create_person,
    save_face_record,
    get_face_record_id,
    save_identity_decision,
    get_identity_decisions
)

from src.image_scanner import scan_images
from src.face_detector import count_faces
from src.face_recognizer import extract_face_embeddings
from src.face_clustering import cluster_faces
from src.identity_review import find_duplicate_group_candidates


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DETECTOR_MODEL = (
    BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_MODEL = (
    BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
)

RECENT_PATHS_FILE = BASE_DIR / "config" / "recent_paths.json"
FACE_PREVIEW_DIR = BASE_DIR / "test" / "face_previews"

MAX_RECENT_PATHS = 10


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="FaceSort AI",
    page_icon="📸",
    layout="wide"
)


# ==========================================
# DATABASE INITIALIZATION
# ==========================================

initialize_database()


# ==========================================
# SESSION STATE
# ==========================================

SESSION_DEFAULTS = {
    "scanned_images": [],
    "face_embeddings": [],
    "face_preview_results": None,
    "clustering_results": None,
    "selected_folder": "",
    "recent_paths": [],
    "folder_path_input": "",
    "recent_folder_selection": "Choose a previous folder...",
    "review_decisions": {},
    "review_candidate_index": 0,
    "navigate_to_people": False,
    "navigation": "Dashboard"
}

for key, default_value in SESSION_DEFAULTS.items():

    if key not in st.session_state:
        st.session_state[key] = default_value


# ==========================================
# RECENT FOLDER MANAGEMENT
# ==========================================

def load_recent_paths():
    """Load previously used photo-folder paths."""

    if not RECENT_PATHS_FILE.exists():
        return []

    try:

        with open(
            RECENT_PATHS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            paths = json.load(file)

        return paths if isinstance(paths, list) else []

    except (json.JSONDecodeError, OSError):
        return []


def save_recent_path(folder_path):
    """Save a folder path, keeping recent paths first."""

    folder_path = str(Path(folder_path))

    paths = load_recent_paths()

    paths = [
        path for path in paths
        if path.lower() != folder_path.lower()
    ]

    paths.insert(0, folder_path)
    paths = paths[:MAX_RECENT_PATHS]

    RECENT_PATHS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RECENT_PATHS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(paths, file, indent=4)

    return paths


def select_previous_folder():
    """Fill the folder input when a previous path is selected."""

    selected = st.session_state.recent_folder_selection

    if selected != "Choose a previous folder...":
        st.session_state.folder_path_input = selected


if not st.session_state.recent_paths:
    st.session_state.recent_paths = load_recent_paths()


# ==========================================
# GENERAL HELPER FUNCTIONS
# ==========================================

def reset_recognition_results():
    """Reset in-memory recognition results."""

    st.session_state.face_embeddings = []
    st.session_state.face_preview_results = None
    st.session_state.clustering_results = None
    st.session_state.review_decisions = {}
    st.session_state.review_candidate_index = 0


def display_face(face, caption=""):
    """
    Display an aligned face.

    OpenCV images are converted from BGR to RGB.
    Original photographs are never modified.
    """

    aligned_face = face.get("aligned_face")

    if aligned_face is not None:

        rgb_face = cv2.cvtColor(
            aligned_face,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            rgb_face,
            caption=caption,
            use_container_width=True
        )

    else:
        st.warning("Face preview unavailable.")


def create_portrait_thumbnail(
    image_path,
    size=(300, 400)
):
    """
    Create a portrait gallery thumbnail without cropping.

    Preserves the complete photograph and its aspect ratio.
    """

    with Image.open(image_path) as image:

        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        image.thumbnail(
            size,
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "RGB",
            size,
            color=(30, 30, 30)
        )

        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2

        canvas.paste(image, (x, y))

    return canvas


def get_face_label(face):
    """Return a readable image and face label."""

    return (
        f"{Path(face['image_path']).name} "
        f"(Face {face['face_number']})"
    )


# ==========================================
# SQLITE RECOGNITION PERSISTENCE
# ==========================================

def save_recognition_to_database(
    faces,
    person_groups
):
    """
    Save generated person records and detected face records.

    Automatic clustering does not assign confirmed identities.
    Faces are stored without person assignments.
    """

    for group in person_groups:

        create_person(
            person_id=group["person_id"]
        )

    for face in faces:

        save_face_record(
            image_path=face["image_path"],
            face_number=face["face_number"],
            embedding=face["embedding"]
        )


# ==========================================
# IDENTITY REVIEW DECISION KEYS
# ==========================================

def get_candidate_decision_key(candidate):
    """
    Generate a stable key for each review candidate.

    GROUP_GROUP:
        Sorted person IDs.

    UNMATCHED_FACE_GROUP:
        Person ID + unmatched face database ID.

    UNMATCHED_FACE_PAIR:
        Sorted database IDs of both unmatched faces.
    """

    candidate_type = candidate["candidate_type"]

    if candidate_type == "GROUP_GROUP":

        people = sorted([
            candidate["person_a"],
            candidate["person_b"]
        ])

        return f"GROUP::{people[0]}::{people[1]}"

    if candidate_type == "UNMATCHED_FACE_GROUP":

        unmatched_face = candidate["unmatched_face"]

        face_record_id = get_face_record_id(
            unmatched_face["image_path"],
            unmatched_face["face_number"]
        )

        if face_record_id is None:
            raise ValueError(
                "Unmatched face was not found in SQLite. "
                "Run face recognition again to register it."
            )

        return (
            f"UNMATCHED_GROUP::{candidate['person_id']}::"
            f"{face_record_id}"
        )

    if candidate_type == "UNMATCHED_FACE_PAIR":

        face_a_id = get_face_record_id(
            candidate["face_a"]["image_path"],
            candidate["face_a"]["face_number"]
        )

        face_b_id = get_face_record_id(
            candidate["face_b"]["image_path"],
            candidate["face_b"]["face_number"]
        )

        if face_a_id is None or face_b_id is None:
            raise ValueError(
                "One or both unmatched faces are missing from SQLite. "
                "Run face recognition again."
            )

        face_a_id, face_b_id = sorted(
            (face_a_id, face_b_id)
        )

        return f"UNMATCHED_PAIR::{face_a_id}::{face_b_id}"

    raise ValueError(
        f"Unsupported candidate type: {candidate_type}"
    )


# ==========================================
# LOAD PERSISTED DECISIONS
# ==========================================

def load_database_decisions():
    """
    Retrieve the latest saved decision for each candidate.

    Repository results are ordered newest first.
    """

    saved_decisions = get_identity_decisions()

    database_decisions = {}

    for saved in saved_decisions:

        candidate_type = saved["candidate_type"]

        if candidate_type == "GROUP_GROUP":

            people = sorted([
                saved["person_a_id"],
                saved["person_b_id"]
            ])

            key = f"GROUP::{people[0]}::{people[1]}"

        elif candidate_type == "UNMATCHED_FACE_GROUP":

            key = (
                f"UNMATCHED_GROUP::{saved['person_a_id']}::"
                f"{saved['face_record_id']}"
            )

        elif candidate_type == "UNMATCHED_FACE_PAIR":

            face_ids = sorted([
                saved["face_record_id"],
                saved["face_record_id_b"]
            ])

            key = f"UNMATCHED_PAIR::{face_ids[0]}::{face_ids[1]}"

        else:
            continue

        # Keep the newest decision for each candidate.
        if key not in database_decisions:
            database_decisions[key] = saved["decision"]

    return database_decisions


# ==========================================
# SAVE HUMAN IDENTITY DECISIONS
# ==========================================

def save_candidate_decision(
    candidate,
    decision
):
    """
    Persist a human review decision in SQLite.

    This records the decision only.
    It never merges people or assigns faces.
    """

    candidate_type = candidate["candidate_type"]

    common_metrics = {
        "maximum_similarity": candidate["maximum_similarity"],
        "average_similarity": candidate["average_similarity"],
        "minimum_similarity": candidate["minimum_similarity"]
    }

    if candidate_type == "GROUP_GROUP":

        save_identity_decision(
            candidate_type="GROUP_GROUP",
            person_a_id=candidate["person_a"],
            person_b_id=candidate["person_b"],
            decision=decision,
            **common_metrics
        )

    elif candidate_type == "UNMATCHED_FACE_GROUP":

        unmatched_face = candidate["unmatched_face"]

        face_record_id = get_face_record_id(
            unmatched_face["image_path"],
            unmatched_face["face_number"]
        )

        if face_record_id is None:
            raise ValueError(
                "Face record not found in SQLite. "
                "Please run face recognition again."
            )

        save_identity_decision(
            candidate_type="UNMATCHED_FACE_GROUP",
            person_a_id=candidate["person_id"],
            face_record_id=face_record_id,
            decision=decision,
            **common_metrics
        )

    elif candidate_type == "UNMATCHED_FACE_PAIR":

        face_a = candidate["face_a"]
        face_b = candidate["face_b"]

        face_a_id = get_face_record_id(
            face_a["image_path"],
            face_a["face_number"]
        )

        face_b_id = get_face_record_id(
            face_b["image_path"],
            face_b["face_number"]
        )

        if face_a_id is None or face_b_id is None:
            raise ValueError(
                "One or both face records are missing from SQLite. "
                "Please run face recognition again."
            )

        save_identity_decision(
            candidate_type="UNMATCHED_FACE_PAIR",
            person_a_id=None,
            person_b_id=None,
            face_record_id=face_a_id,
            face_record_id_b=face_b_id,
            decision=decision,
            **common_metrics
        )

    else:

        raise ValueError(
            f"Unsupported candidate type: {candidate_type}"
        )


# ==========================================
# NAVIGATION STATE
# ==========================================

if st.session_state.navigate_to_people:

    st.session_state.navigation = "People"
    st.session_state.navigate_to_people = False


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📸 FaceSort AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Photo Scanner",
        "People",
        "Identity Review",
        "Settings"
    ],
    key="navigation"
)

st.sidebar.divider()

st.sidebar.caption("Local AI processing")
st.sidebar.caption("Your photographs stay on your computer.")


# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("FaceSort AI")
    st.caption("Your private, AI-powered photo organiser")

    st.divider()

    total_images = len(st.session_state.scanned_images)
    total_faces = len(st.session_state.face_embeddings)

    results = st.session_state.clustering_results

    total_groups = (
        len(results["person_groups"])
        if results else 0
    )

    total_unmatched = (
        len(results["unmatched_faces"])
        if results else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Photos scanned", total_images)
    col2.metric("Faces detected", total_faces)
    col3.metric("People identified", total_groups)
    col4.metric("Unmatched faces", total_unmatched)

    st.divider()

    if total_images == 0:

        st.info(
            "Start by opening Photo Scanner and selecting "
            "your photograph folder."
        )

    elif results is None:

        st.info(
            "Photographs are scanned. Run face recognition "
            "to identify people."
        )

    else:

        st.success("Face recognition completed successfully!")


# ==========================================
# PHOTO SCANNER
# ==========================================

elif page == "Photo Scanner":

    st.title("Photo Scanner")

    st.write(
        "Select the folder containing your photographs."
    )

    st.subheader("Previously Used Folders")

    recent_options = [
        "Choose a previous folder..."
    ] + st.session_state.recent_paths

    st.selectbox(
        "Select a saved folder",
        options=recent_options,
        key="recent_folder_selection",
        on_change=select_previous_folder
    )

    folder_path = st.text_input(
        "Photo folder path",
        key="folder_path_input",
        placeholder=r"C:\Users\naray\Pictures\Camera Roll"
    )

    recursive = st.checkbox(
        "Include subfolders",
        value=True
    )

    if st.button(
        "Scan Photographs",
        type="primary"
    ):

        if not folder_path.strip():

            st.error("Please enter a folder path.")

        else:

            try:

                with st.spinner("Scanning photographs..."):

                    images = scan_images(
                        folder_path,
                        recursive=recursive
                    )

                st.session_state.scanned_images = images
                st.session_state.selected_folder = folder_path

                st.session_state.recent_paths = save_recent_path(
                    folder_path
                )

                reset_recognition_results()

                st.success("Scanning completed!")

            except (FileNotFoundError, NotADirectoryError) as error:

                st.error(str(error))

    images = st.session_state.scanned_images

    if images:

        st.divider()

        st.metric(
            "Total photographs discovered",
            len(images)
        )

        st.subheader("Discovered Photographs")

        columns = st.columns(4)

        for index, image_path in enumerate(images[:20]):

            with columns[index % 4]:

                try:

                    st.image(
                        create_portrait_thumbnail(image_path),
                        caption=image_path.name,
                        use_container_width=True
                    )

                except Exception as error:

                    st.warning(
                        f"Could not preview {image_path.name}: {error}"
                    )

        if len(images) > 20:

            st.caption(
                f"Showing 20 of {len(images)} photographs."
            )

        st.divider()

        # ----------------------------------
        # FACE DETECTION PREVIEWS
        # ----------------------------------

        st.subheader("Face Detection Previews")

        st.write(
            "Generate copies of your photographs with green boxes "
            "around detected faces. Original photographs are never modified."
        )

        st.caption(
            f"Detection threshold: {FACE_DETECTION_THRESHOLD:.2f} · "
            f"Preview folder: {FACE_PREVIEW_DIR}"
        )

        if st.button("Generate Face Previews", key="generate_face_previews"):

            if not DETECTOR_MODEL.exists():

                st.error(f"YuNet model not found: {DETECTOR_MODEL}")

            else:

                try:

                    with st.spinner("Detecting faces and generating preview images..."):

                        preview_results = count_faces(
                            images,
                            DETECTOR_MODEL,
                            FACE_PREVIEW_DIR,
                            score_threshold=FACE_DETECTION_THRESHOLD
                        )

                    st.session_state.face_preview_results = preview_results
                    st.success("Face previews generated successfully!")
                    st.rerun()

                except Exception as error:

                    st.error(f"Face preview generation failed: {error}")

        preview_results = st.session_state.face_preview_results

        if preview_results is not None:

            preview_col1, preview_col2, preview_col3 = st.columns(3)
            preview_col1.metric("Images processed", preview_results["images_processed"])
            preview_col2.metric("Faces detected", preview_results["total_faces"])
            preview_col3.metric("Previews saved", preview_results["previews_saved"])

            if preview_results["unreadable_images"]:
                st.warning(
                    f"{preview_results['unreadable_images']} images could not be read."
                )

            preview_files = sorted(
                path for path in FACE_PREVIEW_DIR.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
            ) if FACE_PREVIEW_DIR.exists() else []

            if preview_files:

                with st.expander("View generated face previews", expanded=True):

                    preview_columns = st.columns(3)

                    for preview_index, preview_path in enumerate(preview_files):

                        with preview_columns[preview_index % 3]:

                            try:
                                st.image(
                                    str(preview_path),
                                    caption=preview_path.name,
                                    use_container_width=True
                                )
                            except Exception as error:
                                st.warning(f"Could not display {preview_path.name}: {error}")

                st.caption(f"Preview images are also saved locally in: {FACE_PREVIEW_DIR}")

            else:
                st.warning("No preview images were saved. Check the image formats and write permissions.")

        st.divider()

        st.subheader("Face Recognition")

        st.write(
            "Run AI face detection and recognition "
            "on the scanned photographs."
        )

        if st.button(
            "Run Face Recognition",
            type="primary"
        ):

            if not DETECTOR_MODEL.exists():

                st.error(
                    f"YuNet model not found: {DETECTOR_MODEL}"
                )

            elif not RECOGNIZER_MODEL.exists():

                st.error(
                    f"SFace model not found: {RECOGNIZER_MODEL}"
                )

            else:

                try:

                    with st.spinner(
                        "Detecting faces and generating embeddings..."
                    ):

                        faces = extract_face_embeddings(
                            images,
                            DETECTOR_MODEL,
                            RECOGNIZER_MODEL,
                            score_threshold=FACE_DETECTION_THRESHOLD
                        )

                    with st.spinner("Grouping identified faces..."):

                        results = cluster_faces(faces)

                    with st.spinner(
                        "Saving local identity records..."
                    ):

                        save_recognition_to_database(
                            faces,
                            results["person_groups"]
                        )

                    st.session_state.face_embeddings = faces
                    st.session_state.clustering_results = results

                    # Existing SQLite decisions remain untouched.
                    st.session_state.review_decisions = {}
                    st.session_state.review_candidate_index = 0

                    st.success(
                        "Face recognition and clustering completed!"
                    )

                    st.session_state.navigate_to_people = True

                    st.rerun()

                except Exception as error:

                    reset_recognition_results()

                    st.error(
                        f"Face recognition failed: {error}"
                    )

    else:

        st.info(
            "No photographs scanned yet. Select a folder "
            "and click Scan Photographs."
        )


# ==========================================
# PEOPLE
# ==========================================

elif page == "People":

    st.title("Identified People")

    results = st.session_state.clustering_results

    if results is None:

        st.info(
            "Run face recognition from Photo Scanner first."
        )

    else:

        person_groups = results["person_groups"]

        st.metric(
            "People identified",
            len(person_groups)
        )

        st.divider()

        if not person_groups:

            st.warning(
                "No person groups were identified."
            )

        for group in person_groups:

            person_id = group["person_id"]
            faces = group["faces"]

            with st.expander(
                f"{person_id} — {len(faces)} appearances",
                expanded=True
            ):

                columns = st.columns(4)

                for index, face in enumerate(faces):

                    with columns[index % 4]:

                        display_face(
                            face,
                            caption=get_face_label(face)
                        )


# ==========================================
# IDENTITY REVIEW
# ==========================================

elif page == "Identity Review":

    st.title("Identity Review")

    st.caption(
        "Review possible identity matches. "
        "No identities are merged automatically."
    )

    results = st.session_state.clustering_results

    if results is None:

        st.info(
            "Run face recognition from Photo Scanner first."
        )

    else:

        person_groups = results["person_groups"]
        unmatched_faces = results["unmatched_faces"]

        # ----------------------------------
        # GENERATE ALL THREE CANDIDATE TYPES
        # ----------------------------------

        candidates = find_duplicate_group_candidates(
            person_groups,
            unmatched_faces
        )

        st.caption(
            f"Review threshold: {REVIEW_MATCH_THRESHOLD:.3f}"
        )

        # ----------------------------------
        # BUILD DECISION KEYS
        # ----------------------------------

        try:

            candidate_keys = {
                id(candidate): get_candidate_decision_key(candidate)
                for candidate in candidates
            }

        except Exception as error:

            st.error(str(error))
            st.stop()

        # ----------------------------------
        # LOAD SAVED DECISIONS
        # ----------------------------------

        database_decisions = load_database_decisions()

        def get_decision(candidate):

            key = candidate_keys[id(candidate)]

            if key in st.session_state.review_decisions:
                return st.session_state.review_decisions[key]

            return database_decisions.get(key, "PENDING")

        # ----------------------------------
        # CANDIDATE SUMMARY
        # ----------------------------------

        st.subheader("Candidate Summary")

        st.metric(
            "Possible identity matches",
            len(candidates)
        )

        summary_rows = []

        for index, candidate in enumerate(candidates, start=1):

            decision = get_decision(candidate)
            candidate_type = candidate["candidate_type"]

            if candidate_type == "GROUP_GROUP":

                type_label = "Group vs Group"

                description = (
                    f"{candidate['person_a']} / "
                    f"{candidate['person_b']}"
                )

            elif candidate_type == "UNMATCHED_FACE_GROUP":

                type_label = "Unmatched vs Group"

                description = (
                    f"{get_face_label(candidate['unmatched_face'])} "
                    f"→ {candidate['person_id']}"
                )

            else:

                type_label = "Unmatched vs Unmatched"

                description = (
                    f"{get_face_label(candidate['face_a'])} "
                    f"↔ {get_face_label(candidate['face_b'])}"
                )

            summary_rows.append({
                "Candidate": index,
                "Type": type_label,
                "Comparison": description,
                "Maximum similarity": round(
                    candidate["maximum_similarity"], 3
                ),
                "Average similarity": round(
                    candidate["average_similarity"], 3
                ),
                "Minimum similarity": round(
                    candidate["minimum_similarity"], 3
                ),
                "Status": decision
            })

        if summary_rows:

            st.dataframe(
                summary_rows,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No possible identity matches found above "
                "the current review threshold."
            )

        # ----------------------------------
        # REVIEW COUNTS
        # ----------------------------------

        pending_candidates = [
            candidate
            for candidate in candidates
            if get_decision(candidate) == "PENDING"
        ]

        reviewed_count = (
            len(candidates) - len(pending_candidates)
        )

        col1, col2 = st.columns(2)

        col1.metric(
            "Awaiting decision",
            len(pending_candidates)
        )

        col2.metric(
            "Reviewed",
            reviewed_count
        )

        st.divider()

        # ----------------------------------
        # REVIEW ONE PENDING CANDIDATE
        # ----------------------------------

        if not pending_candidates:

            if candidates:

                st.success(
                    "You've reviewed all suggested identity matches."
                )

            else:

                st.info(
                    "There are no candidates to review."
                )

        else:

            candidate = pending_candidates[0]

            decision_key = candidate_keys[id(candidate)]

            candidate_type = candidate["candidate_type"]

            best_comparison = candidate["comparisons"][0]

            face_a = best_comparison["face_a"]
            face_b = best_comparison["face_b"]

            # ----------------------------------
            # REVIEW QUESTION AND LABELS
            # ----------------------------------

            if candidate_type == "GROUP_GROUP":

                left_label = candidate["person_a"]
                right_label = candidate["person_b"]

                st.subheader(
                    f"Are {left_label} and {right_label} "
                    "the same person?"
                )

            elif candidate_type == "UNMATCHED_FACE_GROUP":

                left_label = candidate["person_id"]
                right_label = "Unmatched Face"

                st.subheader(
                    f"Does this unmatched face belong to "
                    f"{candidate['person_id']}?"
                )

                st.caption(
                    "The face was not assigned to a person "
                    "during automatic clustering."
                )

            else:

                left_label = "Unmatched Face A"
                right_label = "Unmatched Face B"

                st.subheader(
                    "Do these two unmatched faces belong "
                    "to the same person?"
                )

                st.caption(
                    "Both faces are currently unmatched. "
                    "They were detected in different photographs."
                )

            # ----------------------------------
            # SIMILARITY METRICS
            # ----------------------------------

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Maximum similarity",
                f"{candidate['maximum_similarity']:.3f}"
            )

            metric2.metric(
                "Average similarity",
                f"{candidate['average_similarity']:.3f}"
            )

            metric3.metric(
                "Minimum similarity",
                f"{candidate['minimum_similarity']:.3f}"
            )

            st.divider()

            # ----------------------------------
            # DISPLAY FACE COMPARISON
            # ----------------------------------

            left, right = st.columns(2)

            with left:

                st.subheader(left_label)

                display_face(
                    face_a,
                    caption=get_face_label(face_a)
                )

            with right:

                st.subheader(right_label)

                display_face(
                    face_b,
                    caption=get_face_label(face_b)
                )

            st.caption(
                "Similarity of displayed face pair: "
                f"{best_comparison['similarity']:.3f}"
            )

            st.divider()

            # ----------------------------------
            # HUMAN DECISION
            # ----------------------------------

            st.subheader("Your Decision")

            st.write(
                "Your decision will be saved locally in SQLite "
                "and remain available after restarting the app. "
                "No person groups will be merged automatically."
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                if st.button(
                    "✅ Same Person",
                    type="primary",
                    use_container_width=True,
                    key="identity_same_person"
                ):

                    try:

                        save_candidate_decision(
                            candidate,
                            "SAME_PERSON"
                        )

                        st.session_state.review_decisions[
                            decision_key
                        ] = "SAME_PERSON"

                        st.rerun()

                    except Exception as error:

                        st.error(
                            f"Could not save decision: {error}"
                        )

            with col2:

                if st.button(
                    "❌ Different People",
                    use_container_width=True,
                    key="identity_different_people"
                ):

                    try:

                        save_candidate_decision(
                            candidate,
                            "DIFFERENT_PEOPLE"
                        )

                        st.session_state.review_decisions[
                            decision_key
                        ] = "DIFFERENT_PEOPLE"

                        st.rerun()

                    except Exception as error:

                        st.error(
                            f"Could not save decision: {error}"
                        )

            with col3:

                if st.button(
                    "⏭ Skip",
                    use_container_width=True,
                    key="identity_skip"
                ):

                    try:

                        save_candidate_decision(
                            candidate,
                            "SKIPPED"
                        )

                        st.session_state.review_decisions[
                            decision_key
                        ] = "SKIPPED"

                        st.rerun()

                    except Exception as error:

                        st.error(
                            f"Could not save decision: {error}"
                        )


# ==========================================
# SETTINGS
# ==========================================

elif page == "Settings":

    st.title("Settings")

    st.write("Current application configuration")

    st.metric(
        "Face detection threshold",
        FACE_DETECTION_THRESHOLD
    )

    st.metric(
        "Clustering similarity threshold",
        (
            st.session_state.clustering_results[
                "similarity_threshold"
            ]
            if st.session_state.clustering_results
            else 0.65
        )
    )

    st.metric(
        "Identity review threshold",
        REVIEW_MATCH_THRESHOLD
    )

    st.caption(
        "Thresholds are configured in config/settings.py "
        "and src/face_clustering.py."
    )

    st.divider()

    st.subheader("Local Database")

    st.success("SQLite database initialized successfully.")

    st.caption(
        f"Database location: {DATABASE_PATH}"
    )

    st.caption(
        "Identity review decisions are stored locally."
    )

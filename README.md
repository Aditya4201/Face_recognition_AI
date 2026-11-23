# FaceSort AI

FaceSort AI is a local-first Python photo organiser that detects faces, generates facial embeddings, compares identities, and groups recurring faces without uploading photographs to an external service.

The Streamlit application provides a dashboard, photo scanner, person gallery, identity-review view, and settings view. It uses YuNet for face detection and SFace for face recognition.

## Features

- Recursively scans folders for JPG, JPEG, PNG, WEBP, BMP, TIFF, and TIF images
- Detects faces with OpenCV YuNet
- Generates aligned face previews and 128-dimensional SFace embeddings
- Compares face embeddings with cosine similarity
- Creates visual verification previews and CSV reports
- Groups faces with conservative complete-link clustering
- Prevents faces from the same photograph from entering one person group
- Preserves original photographs and aspect ratios
- Remembers recently selected folders
- Suggests possible duplicate person groups for human review

## Requirements

- Windows 10 or 11
- Python 3.11
- Anaconda or Conda
- Git
- VS Code or another Python-capable editor

## Installation

Create and activate a Conda environment:

```powershell
conda create -n facesort python=3.11 -y
conda activate facesort
```

Install the project dependencies and Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m pip install streamlit
```

Verify OpenCV:

```powershell
python -c "import cv2; print(cv2.__version__)"
```

## Run the Application

From the repository root:

```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

## Application Pages

| Page | Purpose |
|---|---|
| Dashboard | Shows application and processing status |
| Photo Scanner | Selects a folder and starts image processing |
| People | Displays grouped person faces |
| Identity Review | Lists possible duplicate person groups |
| Settings | Displays application configuration |

Gallery portraits use padding rather than cropping, so the complete photograph remains visible. The application creates previews and reports but does not modify source photographs.

## Project Structure

```text
Face_recognition_AI/
├── app.py
├── config/
│   ├── settings.py
│   └── recent_paths.json
├── models/
│   ├── face_detection_yunet_2023mar.onnx
│   └── face_recognition_sface_2021dec.onnx
├── src/
│   ├── image_scanner.py
│   ├── face_detector.py
│   ├── face_recognizer.py
│   ├── face_verifier.py
│   ├── face_clustering.py
│   ├── face_review.py
│   ├── face_diagnostics.py
│   └── identity_review.py
├── tests/
│   ├── test_face_recognizer.py
│   ├── test_face_verifier.py
│   └── test_face_clustering.py
├── test/
│   ├── face_previews/
│   ├── face_verifications/
│   └── cluster_reviews/
├── output/
├── requirements.txt
└── README.md
```

## Module Overview

| Module | Responsibility |
|---|---|
| `image_scanner.py` | Finds supported images, including in subdirectories |
| `face_detector.py` | Detects faces and creates bounding-box previews |
| `face_recognizer.py` | Aligns faces, generates embeddings, and calculates similarity |
| `face_verifier.py` | Builds pairwise comparison reports and visual previews |
| `face_clustering.py` | Builds conservative person groups and identifies unmatched faces |
| `face_review.py` | Generates contact sheets for offline review |
| `identity_review.py` | Finds candidate duplicate person groups |
| `face_diagnostics.py` | Provides detection and recognition diagnostics |
| `app.py` | Implements the Streamlit interface and application flow |

## How Recognition Works

### 1. Scan

`src/image_scanner.py` discovers supported image files and returns their paths. It does not modify or move source images.

### 2. Detect

YuNet detects each face and returns a bounding box, facial landmarks, and a confidence score.

The configured detection threshold is `0.6` in `config/settings.py`.

### 3. Recognise

SFace aligns and processes each detected face to produce a 128-dimensional embedding.

Cosine similarity compares two embeddings:

\[
\text{Similarity}(A,B)=
\frac{A\cdot B}{\|A\|\|B\|}
\]

Higher values indicate greater similarity. Similarity is not a probability of identity, so its thresholds must be evaluated against the photographs being processed.

The reference threshold is `0.363` in `config/settings.py`. Automatic identity candidates use `0.50`; human-review candidates use `0.363`.

### 4. Cluster

The clustering implementation begins with every face in its own cluster and uses complete-link merging.

Two clusters merge only when:

- Every cross-cluster similarity is at least `0.65`
- The two clusters contain no faces from the same photograph

Clusters with at least two faces become person groups. Single-face clusters remain unmatched. This conservative threshold reduces accidental merges, but may split one person across multiple groups.

The identity-review page compares existing person groups. It does not merge groups or change photographs automatically.

## Processing Outputs

Generated artifacts include:

- Detection previews
- Aligned face previews
- Side-by-side verification previews
- Comparison CSV files
- Cluster review contact sheets

The exact output locations are determined by each module or test script. Add generated outputs and private photographs to `.gitignore` when they are not intended for version control.

## Testing

Run commands from the repository root.

Test face recognition and embeddings:

```powershell
python -m tests.test_face_recognizer
```

Test visual verification:

```powershell
python -m tests.test_face_verifier
```

Test face clustering:

```powershell
python -m tests.test_face_clustering
```

## Privacy and Security

- Image scanning, face detection, embedding generation, comparison, and clustering run locally.
- No external face-recognition API is used.
- Photographs and embeddings remain on the local machine.
- The Streamlit server should remain bound to the local loopback interface.
- Source photographs must never be modified or deleted.
- Private photographs, embeddings, identity databases, and generated personal results should not be committed to GitHub.

Do not deploy this application to a public environment without a separate security review.

## Development Status

Implemented:

- Environment and image scanning
- YuNet face detection
- SFace embedding generation
- Cosine similarity comparison
- Visual verification reports
- Complete-link clustering
- Streamlit dashboard and gallery
- Duplicate person-group candidate review

Current limitations:

- Identity fragmentation may require manual review
- Duplicate-group review only compares existing person groups, not unmatched faces
- Persistent identity memory is not implemented
- Automatic photograph organisation is not implemented

Planned work:

- Persistent identity storage
- Unmatched-face-to-group review
- Confirmed identity correction and reuse
- Photo organisation by person
- Incremental processing and caching
- Improved processing performance
- Packaging and application logging

## Development Principles

- Prefer conservative matching over incorrect identity merging.
- Keep original photographs unchanged.
- Preserve uncertainty and expose similarity scores for human review.
- Keep AI modules independent from the Streamlit interface.
- Store local dependencies and model files explicitly.
- Test recognition stages independently before integrating them.

## References

- [OpenCV documentation](https://docs.opencv.org/)
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo)
- [NumPy documentation](https://numpy.org/doc/)
- [Python documentation](https://docs.python.org/3/)

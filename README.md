# Drip Score — AI-Powered Office Fashion Analyzer

Captures a webcam photo, analyzes outfit and body composition using computer vision,
calculates a **Drip Score out of 100**, shows **similar people by body type**
from **Celeb-FBI pose-estimation**, and shows similar looks from the
**DeepFashion dataset (42,537 outfits)** sorted by visual similarity — descending.

---

## How it works

```
Webcam frame
    │
    ├── MediaPipe Pose          → body landmarks
    │                            → body vector:
    │                              {body_shape, shoulder_hip_ratio, torso_leg_ratio, height_bucket}
    │
    ├── MediaPipe Face Mesh     → face landmarks
    │                            → inter-ocular "face unit" normalization
    │                            → skin tone from forehead/cheek patches
    │
    ├── MediaPipe Selfie Seg    → person mask for cleaner color + embedding crops
    │
    ├── K-Means on torso crop   → dominant clothing colors (LAB colorspace)
    │
    ├── Body-profile retrieval  → compare body vector to Celeb-FBI pose-estimation index
    │                            → similar body profiles ranked by similarity score
    │
    └── Embedding model         → outfit embedding (when enabled)
          (default: HF CLIP ViT-B/32, optional: FashionCLIP / Marqo-FashionSigLIP)
            │
            └── FAISS search → Top-20 similar outfits from DeepFashion
                               sorted by cosine similarity (descending)

Score = Fit (20) + Harmony (20) + Season (15) + Grooming (15)
      + Coherence (10) + Fashion retrieval strength (20) = 100
```

### Similarity formulas

- **Similar body profile score**
  - Distance uses weighted differences in:
    - `body_shape` match/mismatch
    - `shoulder_hip_ratio`
    - `torso_leg_ratio`
    - `height_bucket`
    - light regularizers for BMI/age when present
  - Similarity is: `sim = 1 / (1 + distance)` (higher is better)

- **Fashion match score (DeepFashion)**
  - Uses FAISS inner-product search over L2-normalized embeddings
  - Returned `similarity_score` is clipped to `[0, 1]`
  - Higher means visually closer outfit embedding

- **Fashion retrieval pillar in Drip Score (0..20)**
  - Count matches with `similarity_score >= 0.80`
  - `2 points` per qualifying match
  - Capped at `20`

---

## Setup

### 1. Clone / copy this folder
```bash
cd drip_score
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Build the fashion dataset index (one-time)

You have 4 options when prompted on first run, or run manually:

```bash
# Quick demo — 500 items, ~5-10 min
python dataset_builder.py --max 500

# Medium — 3000 items, ~30 min
python dataset_builder.py --max 3000

# Full — all 42k items, ~2-3 hrs (best results)
python dataset_builder.py

# Use Marqo-FashionSigLIP backend explicitly (optional)
python dataset_builder.py --backend marqo_siglip

# Use FashionCLIP backend explicitly (optional)
python dataset_builder.py --backend fashion_clip

# Validate normalizer first (recommended before full run)
python normalizer.py
```

> Keep app embedding backend and dataset index backend aligned. If you switch
> backend (for example `DRIP_EMBED_BACKEND=fashion_clip`), rebuild the FAISS
> index with the same backend.

### 5. Run the app
```bash
python app.py
```

Analyze an uploaded/local image (no webcam):
```bash
python app.py --image /absolute/path/to/photo.jpg
```

Capture one frame from webcam (no GUI window) and run full analysis:
```bash
python app.py --capture
```

### Test commands (copy/paste)

Use project venv first:
```bash
source .venv311/bin/activate
```

#### A) Uploaded image tests

Quick smoke test (no fashion model; fastest):
```bash
DRIP_ENABLE_FASHION_MODEL=0 python app.py --image img4.jpg
```

Uploaded image + strict male body-profile matching:
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

Uploaded image + strict female body-profile matching:
```bash
DRIP_QUERY_GENDER=1 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

Uploaded image + demo Celeb index (faster startup):
```bash
DRIP_CELEB_DEMO_MAX_ROWS=300 DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

Uploaded image + full production-like run:
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

#### B) Camera tests

Interactive live webcam mode (S scan, R reset, D debug, Q quit):
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py
```

One-shot camera capture mode (recommended on macOS stability issues):
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 DRIP_EMBED_SUBPROCESS_TIMEOUT_S=20 python app.py --capture
```

Camera one-shot without fashion retrieval (fastest, no CLIP dependency):
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=0 python app.py --capture
```

#### C) Dataset/index prep commands

Rebuild DeepFashion FAISS index (HF CLIP, quick demo):
```bash
python dataset_builder.py --backend hf_clip --max 500
```

Rebuild DeepFashion FAISS index (HF CLIP, full):
```bash
python dataset_builder.py --backend hf_clip
```

Prefetch Celeb thumbnails quickly:
```bash
python prefetch_celeb_images.py --limit 1000
```

Prefetch all Celeb thumbnails:
```bash
python prefetch_celeb_images.py
```

### Recommended commands for reliable results

### Edge cases and how to run them

#### 1) Filter similar body profiles by gender

Strict male-only body-profile retrieval:
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

Strict female-only body-profile retrieval:
```bash
DRIP_QUERY_GENDER=1 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

No gender filter (mixed candidates allowed):
```bash
DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

#### 2) Fashion embedding/model is slow or hangs

Set subprocess timeout (recommended):
```bash
DRIP_EMBED_SUBPROCESS=1 DRIP_EMBED_SUBPROCESS_TIMEOUT_S=20 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip python app.py --capture
```

Disable fashion retrieval completely (fastest and safest):
```bash
DRIP_ENABLE_FASHION_MODEL=0 python app.py --image img4.jpg
```

#### 3) Camera mode crashes/stability issues on macOS GUI

Use one-shot camera capture mode (no OpenCV window):
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 DRIP_EMBED_SUBPROCESS_TIMEOUT_S=20 python app.py --capture
```

#### 4) Face not detected / poor lighting

If report says "Face not detected", capture with better light and visible face:
```bash
DRIP_ENABLE_FASHION_MODEL=0 python app.py --capture
```
Then retry with fashion enabled once face detection is stable:
```bash
DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --capture
```

#### 5) Quick demo vs full production index

Fast demo (smaller Celeb index):
```bash
DRIP_CELEB_DEMO_MAX_ROWS=300 DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip python app.py --image img4.jpg
```

Full production-like Celeb index:
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip python app.py --image img4.jpg
```

#### 6) Rebuild dataset/index from scratch

Delete old DeepFashion index artifacts:
```bash
rm -f cache/faiss_index.bin cache/metadata.json cache/build_info.json
```

Rebuild quick:
```bash
python dataset_builder.py --backend hf_clip --max 500
```

Rebuild full:
```bash
python dataset_builder.py --backend hf_clip
```

#### 7) Celeb thumbnails missing for some body matches

Prefetch a subset quickly:
```bash
python prefetch_celeb_images.py --limit 1000
```

Prefetch all:
```bash
python prefetch_celeb_images.py
```

#### 8) High fashion score confusion

Fashion retrieval score is based on count of matches with similarity >= 0.80:
- `2 points` per match
- capped at `20`

To inspect this in report JSON, open:
- `score.values.fashion_retrieval`
- `score.explanations.fashion_retrieval`
- `score.calculation.fashion_component`

### Output artifacts (for visualization)

For input `img4.jpg`, these files are generated in `cache/`:
- `result_img4.jpg` → visual report panel
- `report_img4.json` → consolidated machine-readable report
- `body_profile_matches.json` → similar body profiles snapshot
- `outfit_matches/manifest.json` → outfit matches + local thumbnail paths

`report_<image>.json` includes:
- extracted body profile and skin/clothing profiles
- complete score breakdown (pillar max + value + explanation)
- fashion matches with similarity scores
- similar body profiles with similarity scores + thumbnail paths

Uploaded image example (`img4.jpg`) writes:
- `cache/result_img4.jpg`
- `cache/report_img4.json`

Camera one-shot (`--capture`) writes:
- `cache/camera_capture.jpg`
- `cache/result_camera_capture.jpg`
- `cache/report_camera_capture.json`

Enable FashionSigLIP outfit-matching model (optional):
```bash
DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=marqo_siglip python app.py
```
Stable default backend:
```bash
DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip python app.py
```
FashionCLIP backend:
```bash
DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=fashion_clip python app.py
```

If you see native crashes while loading embeddings, keep HF CLIP in isolated
subprocess mode (enabled by default):
```bash
DRIP_EMBED_SUBPROCESS=1 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip python app.py
```
You can use the same isolation mode with FashionCLIP:
```bash
DRIP_EMBED_SUBPROCESS=1 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=fashion_clip python app.py
```

For maximum stability, use a dedicated clean env for FashionCLIP worker
(no mediapipe/opencv in that env):
```bash
python -m venv .venv-fashionclip
source .venv-fashionclip/bin/activate
pip install numpy pillow torch transformers huggingface-hub safetensors tokenizers
deactivate
DRIP_FASHIONCLIP_PYTHON=.venv-fashionclip/bin/python DRIP_EMBED_SUBPROCESS=1 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=fashion_clip python app.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `S` | Scan drip — freeze frame and analyze |
| `R` | Reset — back to live feed |
| `D` | Toggle debug overlay (face unit px, distance) |
| `1`–`9` | Page through fashion match results |
| `Q` | Quit |

---

## Scoring pillars

| Pillar | Points | What it measures |
|--------|--------|-----------------|
| Fit & proportion | 20 | Does outfit suit detected body shape |
| Color harmony | 20 | Monochromatic / complementary / clashing |
| Color season match | 15 | Clothing vs skin tone seasonal palette |
| Grooming | 15 | Hair neatness, collar presence, accessories |
| Style coherence | 10 | Single coherent aesthetic vs mixed signals |
| Fashion retrieval strength | 20 | Number of DeepFashion matches with similarity >= 0.80 (2 points each, capped at 20) |

## Score bands

| Score | Label | Commentary style |
|-------|-------|-----------------|
| 0–30 | NEEDS WORK | "HR wants a word." |
| 31–45 | MEDIOCRE | "Business casual. But which business?" |
| 46–60 | DECENT | "B- energy. You read the memo." |
| 61–75 | SHARP | "The fit is doing the networking for you." |
| 76–85 | ELEVATED | "Your outfit has a 5-year plan." |
| 86–100 | DRIP LORD | "The CEO is taking notes on YOUR fit." |

---

## Perspective normalization

All body measurements are divided by the **inter-ocular distance** (outer eye
corner to outer eye corner in pixels). This makes shoulder/hip ratios identical
whether you stand 1 m or 3 m from the camera.

Test it:
```bash
python normalizer.py
```
The face-unit value should stay within ±5 px as you step closer/farther.

---

## File structure

```
drip_score/
├── app.py                  Main OpenCV app
├── feature_extractor.py    Pose, skin tone, clothing colors, embedding
├── normalizer.py           Perspective-invariant measurement system
├── scorer.py               6-pillar scoring logic (out of 100)
├── body_profile_db.py      Celeb-FBI body-profile retrieval + thumbnail cache
├── vector_db.py            FAISS search over DeepFashion
├── color_theory.py         Seasonal palettes + color harmony
├── commentary.py           Gen-Z score commentary
├── dataset_builder.py      Download + embed DeepFashion dataset
├── requirements.txt
├── cache/
│   ├── faiss_index.bin     FAISS index (built by dataset_builder)
│   └── metadata.json       Item metadata
└── README.md
```

---

## Dataset

**Marqo/deepfashion-multimodal** — 42,537 fashion images with:
- `image` — product photo
- `category1` — men / women
- `category2` — denim, jackets, dresses, etc.
- `text` — natural language outfit description

Embedded using **Marqo-FashionSigLIP** (ViT-B-16-SigLIP fine-tuned on fashion,
best-in-class on fashion retrieval benchmarks).

Body-profile similarity uses:
- `alecccdd/celeb-fbi-pose-estimation` (pose-derived features)
- `alecccdd/celeb-fbi` (thumbnails/biometric metadata)

---

## System requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended for full dataset)
- Webcam
- ~2 GB disk for model + full index
- Internet for first-time model + dataset download

> Important: this project uses MediaPipe `solutions` APIs. If you use Python 3.14
> and see `AttributeError: module 'mediapipe' has no attribute 'solutions'`, create
> a Python 3.10-3.12 virtual environment and reinstall dependencies.

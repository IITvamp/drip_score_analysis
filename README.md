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

### Recommended commands for reliable results

If you are using the local project env:
```bash
source .venv311/bin/activate
```

Single image, **strict male** body-profile matching, fast Celeb demo index:
```bash
DRIP_CELEB_DEMO_MAX_ROWS=300 DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=0 python app.py --image img4.jpg
```

Single image, **strict female** body-profile matching, fast Celeb demo index:
```bash
DRIP_CELEB_DEMO_MAX_ROWS=300 DRIP_QUERY_GENDER=1 DRIP_ENABLE_FASHION_MODEL=0 python app.py --image img4.jpg
```

Single image with fashion retrieval enabled (HF CLIP):
```bash
DRIP_CELEB_DEMO_MAX_ROWS=300 DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

Production-like full Celeb body index (no demo limit):
```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

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

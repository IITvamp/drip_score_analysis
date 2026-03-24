# Drip Score: End-to-End Deep Dive

This document explains the project flow in plain language from input image to final JSON/report output.

---

## 1) What this project does

Given a person image (uploaded or webcam capture), the app:

1. extracts body/face/clothing features
2. computes a Drip Score
3. finds similar body profiles from Celeb-FBI
4. finds similar outfits from DeepFashion
5. writes a visual panel and machine-readable JSON report

Core entry point: `app.py`

---

## 2) Main components

- `app.py`
  - orchestrates the pipeline
  - runs image/webcam mode
  - builds result panel and writes reports

- `feature_extractor.py`
  - body vector from pose landmarks
  - skin tone extraction
  - clothing color extraction
  - fashion image embedding

- `body_profile_db.py`
  - builds/loads Celeb-FBI profile index
  - computes similar body profiles
  - caches matched profile thumbnails locally

- `vector_db.py`
  - loads FAISS index + metadata for DeepFashion
  - returns nearest outfit matches for a query embedding
  - fetches outfit images for the selected matches

- `scorer.py`
  - computes pillar scores
  - computes final total score
  - provides full score calculation trace

- `dataset_builder.py`
  - one-time build of DeepFashion embedding index

- `prefetch_celeb_images.py`
  - optional utility to preload Celeb-FBI images into local cache

---

## 3) Datasets and how they are used

### A) Body-profile retrieval

1. `alecccdd/celeb-fbi-pose-estimation`
   - used to build body-profile feature index
   - stored locally at:
     - `cache/celeb_body_profiles/index.json` (full)
     - `cache/celeb_body_profiles/index_demo_<N>.json` (demo mode)

2. `alecccdd/celeb-fbi`
   - used to fetch profile images (thumbnails)
   - cached at:
     - `cache/celeb_body_profiles/images/<id>.jpg`

### B) Outfit retrieval

1. `Marqo/deepfashion-multimodal`
   - used by `dataset_builder.py` to create:
     - `cache/faiss_index.bin` (vector index)
     - `cache/metadata.json` (item metadata map)
     - `cache/build_info.json`
   - during runtime, nearest outfit matches come from FAISS + metadata.

---

## 4) Models used

### CV feature extraction

- MediaPipe Pose
  - body landmarks
- MediaPipe Face Mesh
  - face landmarks, normalization
- MediaPipe Selfie Segmentation
  - person mask

### Fashion embedding backends

- default: HF CLIP (`openai/clip-vit-base-patch32`)
- optional:
  - FashionCLIP (`patrickjohncyh/fashion-clip`)
  - Marqo FashionSigLIP

Runtime backend is controlled by `DRIP_EMBED_BACKEND`.

---

## 5) Feature extraction details

### Body vector (from pose + normalizer)

Computed fields include:

- `shoulder_width_fu`
- `hip_width_fu`
- `torso_height_fu`
- `leg_height_fu`
- `shoulder_hip_ratio`
- `hip_waist_ratio`
- `torso_leg_ratio`
- `body_shape`
- `height_bucket`

`*_fu` means face-unit normalized (camera-distance robust).

### Skin profile

- sampled from forehead + cheek patches
- converted to LAB
- seasonal classification (`spring/summer/autumn/winter`)

### Clothing profile

- dominant colors from torso + lower body via KMeans
- LAB conversion
- distinct-color count

---

## 6) Similar body profile math

For each candidate profile:

1. compute weighted distance on:
   - shape mismatch
   - shoulder/hip ratio difference
   - torso/leg ratio difference
   - height bucket difference
   - small BMI/age regularizers (if present)

2. convert to similarity:

`similarity = 1 / (1 + distance)`

3. rank descending by similarity.

### Strict gender filtering

Set query gender:

- `DRIP_QUERY_GENDER=0` (male)
- `DRIP_QUERY_GENDER=1` (female)

In strict mode, only matching-gender candidates are considered.

---

## 7) Outfit retrieval math

1. query image -> fashion embedding
2. embedding compared against FAISS index vectors
3. top matches returned with `similarity_score` in `[0,1]`
4. top matched outfit images fetched and cached under:
   - `cache/outfit_matches/*.jpg`
   - `cache/outfit_matches/manifest.json`

Current runtime fetch size is capped to 10 outfits.

---

## 8) Drip Score calculation (current)

Pillars:

- Fit (20)
- Harmony (20)
- Season (15)
- Grooming (15)
- Coherence (10)
- Fashion retrieval pillar (20)

Final total now uses hybrid math on the 5 core normalized pillars:

1. Mahalanobis base score against an ideal vector with correlations
2. multiplicative fit gate: `fit^1.8`
3. hard veto: if `fit < 0.3`, total max is 25

Detailed trace is written into report JSON under:

- `score.calculation`

Diagnostics:

- `score.diagnostics.outfit_match_delta`

---

## 9) Outputs generated per run

For input `img4.jpg`, typical outputs:

- `cache/result_img4.jpg` (visual card)
- `cache/report_img4.json` (full structured report)
- `cache/body_profile_matches.json`
- `cache/outfit_matches/manifest.json`

`report_*.json` includes:

- input settings/env context
- body/skin/clothing extracted profiles
- score values + explanations + full calculation trace
- similar body profiles with IDs/similarity/image paths
- fashion matches with similarity/image paths

---

## 10) Common commands

### Environment

```bash
cd /Users/vamp/Documents/omaverse/drip_score
source .venv311/bin/activate
```

### Build/rebuild DeepFashion index (hf_clip)

```bash
rm -f cache/faiss_index.bin cache/metadata.json cache/build_info.json
python dataset_builder.py --max 5000 --backend hf_clip
```

### Prefetch Celeb-FBI images

```bash
python prefetch_celeb_images.py
```

### Single image run (strict male + fashion enabled)

```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py --image img4.jpg
```

### Webcam run

```bash
DRIP_QUERY_GENDER=0 DRIP_ENABLE_FASHION_MODEL=1 DRIP_EMBED_BACKEND=hf_clip DRIP_EMBED_SUBPROCESS=1 python app.py
```

---

## 11) Why reruns can feel expensive

- New `python app.py ...` process means feature extraction runs again.
- With subprocess embedding mode, CLIP load/embedding happens in isolated child process per run.
- If index build was interrupted, body index may rebuild next run.

---

## 12) Known practical caveats

- OpenCV + MediaPipe can print duplicate-class warnings on macOS.
- Hugging Face fetch retries may appear after success logs; outputs can still be valid.
- Some remote images can fail first fetch; cached reruns usually improve coverage.

---

If you want, next step is to add a simple architecture diagram (`.md` + Mermaid) and a one-page "operations runbook" for your team.


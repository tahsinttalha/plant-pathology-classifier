# Plant Pathology Classifier

A production-grade computer vision inference service for 38-class plant leaf disease classification, engineered with a decoupled ONNX Runtime inference engine (exported losslessly from the trained Keras CNN), FastAPI REST service, configurable confidence thresholding, and a minimalist web interface ready for local, Docker, or free GitHub-to-Vercel deployment.

---

## 1. Overview & Business Problem

Standard closed-set image classifiers always emit a top-1 class label via softmax argmax—even when presented with blurry mobile captures, occluded foliage, or non-leaf images. In precision agriculture and smallholder crop advisory workflows, **uncalibrated false-positive diagnoses carry direct operational costs**: misdiagnosing healthy tissue or an ambiguous photo can lead to unnecessary fungicide/pesticide application, wasted input spend, or premature crop culling.

This system mitigates forced misclassifications by decoupling inference from the presentation layer and enforcing an explicit **confidence rejection gate** (default threshold `0.75`):

- **Confident (`p_max >= threshold`):** Emits the predicted crop species and pathology label alongside ranked alternatives and inference latency.
- **Uncertain (`p_max < threshold`):** Withholds the disease label, returns `"status": "uncertain"`, and prompts the operator with `"Uncertain — upload a clearer, single-leaf photograph"` rather than fabricating a diagnosis.

---

## 2. System Architecture

```text
+-----------------------------------+        +-----------------------------------+
|  Browser UI (static/)             |        |  External REST / Mobile Clients   |
|  Minimal Composer + Report Flow   |        |  multipart/form-data + Google URL |
+-----------------+-----------------+        +-----------------+-----------------+
                  |                                            |
                  |  GET / , GET /health , POST /predict , POST /predict/drive
                  +---------------------+----------------------+
                                        |
                                        v
                   +----------------------------------------+
                   |           FastAPI Service              |
                   |               (api.py)                 |
                   |  - Lifespan singleton model loading    |
                   |  - MIME, Drive URL & PIL validation    |
                   |  - Async threadpool dispatch           |
                   +--------------------+-------------------+
                                        |
                                        v
                   +----------------------------------------+
                   |        PlantDiseasePredictor           |
                   |            (inference.py)              |
                   |  - RGB conversion & 128x128 bilinear   |
                   |  - Unscaled float32 [0.0, 255.0]       |
                   |  - Confidence thresholding (0.75 gate) |
                   |  - High-resolution perf_counter timing |
                   +--------------------+-------------------+
                                        |
                                        v
                   +----------------------------------------+
                   |      ONNX Runtime CNN Inference        |
                   |      (plant_disease_model.onnx)        |
                   |  - Exported from plant_disease_model.h5|
                   |  - 38-class softmax (class_names.json) |
                   +----------------------------------------+
```

---

## 3. Model & Preprocessing Specifications

| Property | Specification |
| :--- | :--- |
| **Dataset** | PlantVillage (70,295 training images / 17,117 validation images across 38 classes) |
| **Architecture** | 5-block Sequential CNN (`32 -> 64 -> 128 -> 256 -> 512` double `Conv2D(3x3, ReLU)` + `MaxPool2D(2x2)`), `Flatten`, `Dense(1500, ReLU)`, `Dropout(0.4)`, `Dense(38, Softmax)` |
| **Optimization** | `Adam(learning_rate=1e-4)`, categorical cross-entropy, 10 epochs |
| **Recorded Metrics** | Training accuracy `98.36%` (`loss: 0.0508`) / Validation accuracy `95.95%` (`val_loss: 0.1441`; peak val accuracy `96.15%` at epoch 6) |
| **Input Contract** | `128×128` RGB, bilinear resize, `np.float32` tensor in **`[0.0, 255.0]`** (no `/ 255.0` rescaling; matches Keras `image_dataset_from_directory`) |
| **Production Artifact** | `plant_disease_model.onnx` (`29.93 MB`, exported directly from `plant_disease_model.h5` with `<= 3.28e-07` max probability delta) |
| **Default Gate** | `0.75` minimum top-1 softmax probability (`0.0`–`1.0` configurable per request) |

---

## 4. Quickstart (Local, Docker & Vercel via GitHub)

### Local Setup (Python 3.11+)

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser for the web interface, or `http://127.0.0.1:8000/docs` for interactive OpenAPI documentation.

### Deploy for Free on Vercel via GitHub

Because inference uses `onnxruntime` + `plant_disease_model.onnx` (`29.93 MB`) and excludes `plant_disease_model.h5` and `tensorflow` via `.vercelignore` and `vercel.json`, the full Python FastAPI + frontend bundle stays at `~206 MB` unzipped (under Vercel's `250 MB` limit):

1. Commit and push this repository to GitHub (`git add . && git commit -m "Refactor to FastAPI + ONNX" && git push`).
2. In the Vercel dashboard, click **Add New → Project** and import your GitHub repository.
3. Keep the default settings (Vercel reads `pyproject.toml`, `vercel.json`, and `requirements.txt` automatically) and click **Deploy**.

### Docker Deployment

```bash
docker build -t plant-pathology-api .
docker run --rm -p 8000:8000 plant-pathology-api
```

### API Usage (`curl`)

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Run a leaf prediction with an explicit confidence threshold:

```bash
curl -X POST "http://127.0.0.1:8000/predict?threshold=0.75" \
  -H "accept: application/json" \
  -F "file=@sample_leaf.jpg;type=image/jpeg"
```

Example response:

```json
{
  "status": "confident",
  "diagnosis": "Tomato Leaf: Early_blight",
  "confidence": 0.9614,
  "threshold": 0.75,
  "latency_ms": 18.42,
  "top_predictions": [
    { "label": "Tomato Leaf: Early_blight", "confidence": 0.9614 },
    { "label": "Tomato Leaf: Late_blight", "confidence": 0.0241 },
    { "label": "Tomato Leaf: Target_Spot", "confidence": 0.0089 }
  ]
}
```

### Automated Verification & ONNX Re-export

```bash
pip install -r requirements-dev.txt
pytest -v

# Optional: re-export plant_disease_model.onnx from plant_disease_model.h5
python scripts/export_onnx.py
```

---

## 5. Known Limitations & Failure Modes

1. **Lab-to-Field Domain Shift:** PlantVillage imagery consists predominantly of detached single leaves photographed under uniform lighting against neutral studio backgrounds. Accuracy degrades when evaluating leaves in cluttered outdoor canopies, harsh shadows, specular glare, or soil-heavy backgrounds.
2. **Single-Leaf Framing Assumption:** The `128×128` input resolution discards fine lesion texture when an entire plant or wide field canopy is photographed. Specimens must fill the frame with a single leaf.
3. **Closed-Set Softmax Overconfidence:** Confidence thresholding reduces low-margin errors, but deep ReLU networks trained without explicit out-of-distribution (OOD) calibration can still assign high softmax probability to adversarial or severely shifted non-plant textures. Production field deployment should pair this service with temperature scaling, a dedicated leaf-detector pre-filter, and field-collected fine-tuning data.
4. **Co-Infections & Unseen Crops:** Because the final layer uses single-label softmax over 38 fixed PlantVillage categories, simultaneous co-infections on one leaf or species outside the 38 classes cannot be natively represented.

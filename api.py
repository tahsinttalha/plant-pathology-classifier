"""Production FastAPI service for the Plant Pathology Classifier."""

from __future__ import annotations

import io
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from inference import DEFAULT_THRESHOLD, PlantDiseasePredictor

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = BASE_DIR / "assets"
FRONTEND_DIR = STATIC_DIR if STATIC_DIR.is_dir() else ASSETS_DIR
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)
MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB safety guard

_DRIVE_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^https://drive\.google\.com/file/d/([A-Za-z0-9_-]{10,128})(?:/|\?|$)"
    ),
    re.compile(
        r"^https://drive\.google\.com/(?:open|uc)\?(?:[^#]*&)?id=([A-Za-z0-9_-]{10,128})(?:&|$)"
    ),
    re.compile(
        r"^https://lh3\.googleusercontent\.com/d/([A-Za-z0-9_-]{10,128})(?:=|\?|/|$)"
    ),
)


class DrivePredictRequest(BaseModel):
    """Payload for importing a shared Google Drive leaf image."""

    url: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Instantiate the inference predictor once at application startup."""
    app.state.predictor = PlantDiseasePredictor(
        class_names_path=BASE_DIR / "class_names.json",
        default_threshold=DEFAULT_THRESHOLD,
    )
    yield
    app.state.predictor = None


app = FastAPI(
    title="Plant Pathology Classifier API",
    description=(
        "Production REST API for 38-class plant leaf disease classification "
        "with calibrated confidence thresholding."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

if FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _get_predictor(request: Request) -> PlantDiseasePredictor:
    predictor: PlantDiseasePredictor | None = getattr(
        request.app.state, "predictor", None
    )
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Inference model is not loaded.",
        )
    return predictor


def _decode_pil_image(payload: bytes) -> Image.Image:
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image exceeds the 15 MB limit.",
        )
    try:
        with Image.open(io.BytesIO(payload)) as img:
            img.load()
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid or corrupted image file.",
        ) from exc


def _extract_drive_file_id(url: str) -> str | None:
    cleaned = url.strip()
    for pattern in _DRIVE_ID_PATTERNS:
        match = pattern.match(cleaned)
        if match:
            return match.group(1)
    return None


@app.get("/", include_in_schema=False)
async def serve_index() -> FileResponse:
    """Serve the redesigned minimalist diagnostic web interface."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=404, detail="Frontend assets not found.")
    return FileResponse(index_path)


@app.get("/health")
async def health_check(request: Request) -> dict[str, Any]:
    """Return service liveness and loaded model metadata."""
    predictor: PlantDiseasePredictor | None = getattr(
        request.app.state, "predictor", None
    )
    loaded = predictor is not None and predictor.is_loaded
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "num_classes": predictor.num_classes if loaded else 0,
        "threshold": predictor.default_threshold if loaded else DEFAULT_THRESHOLD,
    }


@app.post("/predict")
async def predict_leaf(
    request: Request,
    file: UploadFile = File(...),
    threshold: float = Query(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to emit a disease diagnosis.",
    ),
) -> dict[str, Any]:
    """Classify an uploaded leaf image with confidence thresholding."""
    predictor = _get_predictor(request)

    content_type = (file.content_type or "").split(";")[0].strip().lower()
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported media type '{content_type or 'unknown'}'. "
                "Upload a JPEG, PNG, or WEBP image."
            ),
        )

    payload = await file.read()
    image = _decode_pil_image(payload)
    return await run_in_threadpool(predictor.predict, image, threshold)


@app.post("/predict/drive")
async def predict_from_drive(
    request: Request,
    body: DrivePredictRequest,
    threshold: float = Query(
        default=DEFAULT_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to emit a disease diagnosis.",
    ),
) -> dict[str, Any]:
    """Fetch a publicly shared Google Drive image and classify it."""
    predictor = _get_predictor(request)

    file_id = _extract_drive_file_id(body.url)
    if not file_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Enter a valid Google Drive file share URL "
                "(e.g. https://drive.google.com/file/d/<id>/view)."
            ),
        )

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(download_url)
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to reach Google Drive to retrieve the image.",
        ) from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=400,
            detail=(
                "Google Drive file could not be downloaded. Ensure sharing is "
                "set to 'Anyone with the link'."
            ),
        )

    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type and content_type.startswith("text/html"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Google Drive returned a webpage instead of an image. Verify the "
                "link points to a public image file."
            ),
        )

    image = _decode_pil_image(resp.content)
    result = await run_in_threadpool(predictor.predict, image, threshold)
    result["image_dimensions"] = f"{image.width}×{image.height}"
    return result

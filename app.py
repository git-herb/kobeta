import base64
import json
import os
import re
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import pipeline

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
# 산출물 파일명 규칙: (파일명)_(진행단계)_(산출물명)_(시간).ext
ENCODED = re.compile(r"^(?P<base>.+)_(?P<step>\d+)_(?P<label>[A-Za-z0-9]+)_(?P<ts>\d{8}-\d{6})\.(?P<ext>[A-Za-z0-9]+)$")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "output"


def _classify(filename: str) -> tuple[str | None, str | None]:
    """업로드 파일의 역할과 base 추정. 역할 ∈ {video, audio, original, corrected, metadata, frame}."""
    p = Path(filename)
    ext = p.suffix.lower()
    if ext in VIDEO_EXT:
        return "video", _sanitize(p.stem)
    m = ENCODED.match(p.name)
    if m:
        label = m.group("label").lower()
        base = m.group("base")
        if label.startswith("frame"):
            return "frame", base
        if label in {"audio", "original", "corrected", "metadata", "tech"}:
            return label, base
    # 규칙에 맞지 않는 파일 폴백 (확장자 기준)
    if ext == ".mp3":
        return "audio", None
    if ext == ".json":
        return "metadata", None
    if ext == ".srt":
        return ("original" if "original" in p.stem.lower() else "corrected"), None
    return None, None

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    sys.exit("[오류] OPENAI_API_KEY가 설정되지 않았습니다. .env.example을 .env로 복사한 뒤 키를 입력하세요.")

if not shutil.which("ffmpeg"):
    sys.exit("[오류] ffmpeg를 찾을 수 없습니다. 설치 후 PATH에 추가하세요. (예: winget install Gyan.FFmpeg)")

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
JOBS_DIR = BASE_DIR / "jobs"

app = FastAPI()
JOBS: dict[str, dict] = {}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


def _save(uf: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as f:
        shutil.copyfileobj(uf.file, f)


@app.post("/upload")
async def upload(background: BackgroundTasks,
                 files: list[UploadFile],
                 whisper_backend: str = Form("openai"),
                 whisper_model: str = Form("base"),
                 llm_model: str = Form("gpt-4o-mini")):
    """영상 또는 기존 산출물(여러 개)을 받아 분석을 시작/재개한다.
    이미 있는 산출물 단계는 건너뛰고 없는 단계만 수행한다."""
    job_id = uuid.uuid4().hex
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    video_path: Path | None = None
    provided: dict[str, Path] = {}
    base: str | None = None
    for uf in files:
        role, b = _classify(uf.filename or "")
        if role is None:
            continue
        base = base or b
        if role == "video":
            video_path = job_dir / f"input{Path(uf.filename).suffix.lower()}"
            _save(uf, video_path)
        elif role == "frame":
            (job_dir / "frames").mkdir(exist_ok=True)
            _save(uf, job_dir / "frames" / Path(uf.filename).name)
        else:
            dest = job_dir / Path(uf.filename).name
            _save(uf, dest)
            provided[role] = dest

    if video_path is None and not provided:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="영상 또는 산출물 파일이 필요합니다.")

    base = base or "output"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    options = {"whisper_backend": whisper_backend, "whisper_model": whisper_model, "llm_model": llm_model}
    JOBS[job_id] = {"status": "queued", "step": 0, "step_name": pipeline.STEP_NAMES[0],
                    "error": None, "warnings": [], "scene_total": 0, "scene_done": 0}
    background.add_task(pipeline.run_pipeline, job_id, JOBS, job_dir, base, ts, video_path, provided, options)
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="존재하지 않는 작업입니다.")
    return {"status": job["status"], "step": job["step"], "step_name": job["step_name"],
            "total_steps": len(pipeline.STEP_NAMES) - 1, "error": job["error"],
            "warnings": job["warnings"],
            "scene_total": job.get("scene_total", 0), "scene_done": job.get("scene_done", 0)}


def _data_uri(path: Path, mime: str) -> str | None:
    if not path.is_file():
        return None
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


@app.post("/jobs/{job_id}/open")
def open_folder(job_id: str):
    """산출물이 저장된 작업 폴더를 탐색기로 연다 (localhost 전용)."""
    job_dir = (JOBS_DIR / job_id).resolve()
    if JOBS_DIR.resolve() not in job_dir.parents or not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="작업 폴더를 찾을 수 없습니다.")
    os.startfile(job_dir)  # Windows 탐색기로 열기
    return {"ok": True}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    """완료된 작업의 산출물 '내용'을 반환한다 (파일 다운로드 없이 UI에서 바로 표출).
    프레임·오디오는 base64 data URI로 인라인 포함한다."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="존재하지 않는 작업입니다.")
    if job["status"] != "done":
        raise HTTPException(status_code=409, detail="아직 완료되지 않은 작업입니다.")

    job_dir = JOBS_DIR / job_id
    manifest = json.loads((job_dir / "manifest.json").read_text(encoding="utf-8"))
    art = manifest["artifacts"]

    def _read(name: str | None) -> str:
        if not name or not (job_dir / name).is_file():
            return ""
        return (job_dir / name).read_text(encoding="utf-8")

    metadata = json.loads(_read(art.get("metadata")) or "{}")
    for scene in metadata.get("scenes", []):
        scene["frame_data"] = _data_uri(job_dir / scene["frame"], "image/jpeg")
    tech = json.loads(_read(art.get("tech")) or "null")
    return {
        "original_srt": _read(art.get("original")),
        "corrected_srt": _read(art.get("corrected")),
        "audio_data": _data_uri(job_dir / art["audio"], "audio/mpeg") if art.get("audio") else None,
        "metadata": metadata,
        "tech": tech,
    }

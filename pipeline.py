"""영상 분석 파이프라인 6단계. 각 단계는 jobs/<job_id>/ 안의 파일을 입출력으로 사용한다."""
import base64
import json
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import srt as srtlib
from dotenv import load_dotenv
from openai import OpenAI
from srt import Subtitle

import srt_utils

TRANSCRIBE_MODEL = "whisper-1"
TEXT_MODEL = "gpt-4o"
VISION_MODEL = "gpt-4o-mini"
AUDIO_LIMIT_BYTES = 24 * 1024 * 1024  # Whisper API 25MB 제한에 1MB 안전 마진
AUDIO_BYTES_PER_SEC = 8000  # 64kbps mono mp3
SCENE_WORKERS = 8  # ⑤ 장면 분석 병렬 처리 동시 실행 수 (I/O 바운드)

load_dotenv()
client = OpenAI()


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run(["ffmpeg", "-y", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패: {result.stderr.strip().splitlines()[-1] if result.stderr else args}")


def probe_duration(media_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(media_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 실패: {media_path}")
    return float(result.stdout.strip())


# ① 오디오 추출
def _has_audio_stream(video_path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def extract_audio(video_path: Path, audio_path: Path) -> None:
    if not _has_audio_stream(video_path):
        raise RuntimeError("영상에 오디오 트랙이 없습니다. 음성이 포함된 영상을 업로드하세요.")
    _run_ffmpeg(["-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)])


def split_audio(audio_path: Path, limit_bytes: int = AUDIO_LIMIT_BYTES) -> list[tuple[Path, float]]:
    """제한 초과 시 시간 기준 분할. (청크 경로, 시작 오프셋 초) 리스트 반환."""
    if audio_path.stat().st_size <= limit_bytes:
        return [(audio_path, 0.0)]
    chunk_sec = limit_bytes // AUDIO_BYTES_PER_SEC
    duration = probe_duration(audio_path)
    chunks = []
    offset = 0
    while offset < duration:
        part = audio_path.with_name(f"audio_part{len(chunks):02d}.mp3")
        _run_ffmpeg(["-ss", str(offset), "-t", str(chunk_sec), "-i", str(audio_path), "-c", "copy", str(part)])
        chunks.append((part, float(offset)))
        offset += chunk_sec
    return chunks


# ② Whisper 자막
def transcribe(chunks: list[tuple[Path, float]]) -> str:
    """청크별 Whisper 호출 → 오프셋 시프트 → 병합된 SRT 텍스트 반환."""
    sub_lists = []
    for path, offset in chunks:
        with open(path, "rb") as f:
            srt_text = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL, file=f, response_format="srt"
            )
        subs = srt_utils.parse(srt_text)
        if offset:
            subs = srt_utils.shift(subs, offset)
        sub_lists.append(subs)
    return srt_utils.compose(srt_utils.merge_and_renumber(sub_lists))


# ②-local: 로컬 faster-whisper (모델 다운로드 후 캐시, CPU int8). 25MB 분할 불필요.
_local_models: dict = {}


def transcribe_local(audio_path: Path, model_size: str) -> str:
    from faster_whisper import WhisperModel

    model = _local_models.get(model_size)
    if model is None:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _local_models[model_size] = model
    segments, _ = model.transcribe(str(audio_path))
    subs = [
        Subtitle(index=i, start=timedelta(seconds=seg.start), end=timedelta(seconds=seg.end),
                 content=seg.text.strip())
        for i, seg in enumerate(segments, start=1)
    ]
    return srt_utils.compose(subs)


# ③ GPT 자막 교정 — 타임코드는 GPT로 보내지 않고, 텍스트만 교정해 원본 큐에 재조립한다.
CORRECTION_BATCH_SIZE = 50
CORRECTION_PROMPT = (
    "다음은 음성 인식(Whisper)으로 생성된 영상 자막의 텍스트 목록이다. 음성 인식 특유의 오류를 교정하라:\n"
    "- 문맥상 어색하게 끼어든 외국어 단어나 무의미한 단어는 문맥에 맞게 고치거나 제거\n"
    "- 발음이 비슷한 다른 단어로 잘못 인식된 부분은 앞뒤 항목의 문맥을 참고해 자연스러운 단어로 교정\n"
    "- 오탈자, 맞춤법, 띄어쓰기, 구두점 교정\n"
    "금지: 번역, 의역, 요약, 새 내용 추가, 항목의 병합·분할·재배열·삭제. 원문 언어 유지. "
    "대사가 없는 항목(예: '...')은 그대로 둔다.\n"
    '정확히 {n}개의 항목을 입력과 같은 순서로 {{"lines": ["...", ...]}} 형식의 JSON으로 반환하라.'
)


def _correct_batch(lines: list[str], model: str) -> tuple[list[str], str | None]:
    """한 배치 교정. 실패 시 (원문, 경고 메시지) 반환 — 잡 전체를 실패시키지 않는다."""
    for _ in range(2):  # 1회 재시도
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": CORRECTION_PROMPT.format(n=len(lines))},
                {"role": "user", "content": json.dumps({"lines": lines}, ensure_ascii=False)},
            ],
        )
        try:
            corrected = json.loads(resp.choices[0].message.content)["lines"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if isinstance(corrected, list) and len(corrected) == len(lines) and all(isinstance(x, str) for x in corrected):
            return corrected, None
    return lines, "교정 응답이 형식에 맞지 않아 원문을 유지했습니다."


def correct_subtitles(original_text: str, model: str = TEXT_MODEL) -> tuple[str, list[str]]:
    """교정된 SRT 텍스트와 경고 목록 반환. 타임코드는 원본과 반드시 동일하다."""
    subs = srt_utils.parse(original_text)
    corrected_texts: list[str] = []
    warnings: list[str] = []
    for i in range(0, len(subs), CORRECTION_BATCH_SIZE):
        batch = [s.content for s in subs[i:i + CORRECTION_BATCH_SIZE]]
        fixed, warn = _correct_batch(batch, model)
        if warn:
            warnings.append(f"배치 {i // CORRECTION_BATCH_SIZE + 1}: {warn}")
        corrected_texts.extend(fixed)
    corrected_subs = [
        Subtitle(index=s.index, start=s.start, end=s.end, content=text)
        for s, text in zip(subs, corrected_texts)
    ]
    corrected_text = srt_utils.compose(corrected_subs)
    if srt_utils.timecode_lines(original_text) != srt_utils.timecode_lines(corrected_text):
        raise RuntimeError("교정 후 타임코드가 변경되었습니다 — 내부 오류")
    return corrected_text, warnings


# ④ 장면 감지
def detect_scenes(video_path: Path) -> list[tuple[float, float]]:
    """(시작 초, 끝 초) 리스트 반환. 컷이 없으면 전체를 1개 장면으로 처리."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video_path), ContentDetector())
    if not scenes:
        return [(0.0, probe_duration(video_path))]
    return [(start.get_seconds(), end.get_seconds()) for start, end in scenes]


# ⑤ 장면별 프레임 추출 + GPT 비전 분석
SCENE_ANALYSIS_PROMPT = (
    "영상의 한 장면을 캡처한 대표 프레임 이미지와 해당 구간의 자막이 주어진다. 이 장면을 분석하라.\n"
    '{"summary": "장면 요약 한두 문장", "visual_description": "화면에 보이는 시각적 묘사", '
    '"keywords": ["핵심 키워드", "..."]} 형식의 JSON으로만 답하라. 한국어로 작성한다.'
)


def extract_frame(video_path: Path, sec: float, frame_path: Path) -> None:
    _run_ffmpeg(["-ss", str(sec), "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(frame_path)])


def analyze_scene(frame_path: Path, subtitle_text: str, model: str = VISION_MODEL) -> dict:
    """프레임 + 자막을 GPT 비전으로 분석. 실패 시 빈 분석 결과 반환(잡을 중단하지 않음)."""
    b64 = base64.b64encode(frame_path.read_bytes()).decode()
    user_text = f"자막:\n{subtitle_text or '(대사 없음)'}"
    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SCENE_ANALYSIS_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "summary": str(data.get("summary", "")),
            "visual_description": str(data.get("visual_description", "")),
            "keywords": [str(k) for k in data.get("keywords", []) if isinstance(k, (str, int, float))],
        }
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return {"summary": "", "visual_description": "", "keywords": []}


def _fmt(sec: float) -> str:
    return srtlib.timedelta_to_srt_timestamp(timedelta(seconds=sec))


# ⑦ 기술 검토: 블랙 구간 / 무음 구간 / 정지 화면 / 클리핑 (ffmpeg 필터, 4종 병렬)
def _ffmpeg_stderr(args: list[str]) -> str:
    r = subprocess.run(["ffmpeg", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stderr or ""


def _detect_black(video_path: Path) -> list[dict]:
    out = _ffmpeg_stderr(["-i", str(video_path), "-vf", "blackdetect=d=0.5:pic_th=0.98", "-an", "-f", "null", "-"])
    return [{"start": float(m[0]), "end": float(m[1]), "duration": float(m[2])}
            for m in re.findall(r"black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)", out)]


def _detect_silence(media_path: Path) -> list[dict]:
    out = _ffmpeg_stderr(["-i", str(media_path), "-af", "silencedetect=noise=-30dB:d=0.5", "-f", "null", "-"])
    starts = re.findall(r"silence_start: ([\d.]+)", out)
    ends = re.findall(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)", out)
    res = []
    for i, (end, dur) in enumerate(ends):
        res.append({"start": float(starts[i]) if i < len(starts) else None,
                    "end": float(end), "duration": float(dur)})
    return res


def _detect_freeze(video_path: Path) -> list[dict]:
    out = _ffmpeg_stderr(["-i", str(video_path), "-vf", "freezedetect=n=-60dB:d=0.5", "-an", "-f", "null", "-"])
    starts = re.findall(r"freeze_start: ([\d.]+)", out)
    durs = re.findall(r"freeze_duration: ([\d.]+)", out)
    ends = re.findall(r"freeze_end: ([\d.]+)", out)
    res = []
    for i in range(len(starts)):
        res.append({"start": float(starts[i]),
                    "end": float(ends[i]) if i < len(ends) else None,
                    "duration": float(durs[i]) if i < len(durs) else None})
    return res


def _detect_clipping(media_path: Path) -> dict:
    out = _ffmpeg_stderr(["-i", str(media_path), "-af", "volumedetect", "-f", "null", "-"])
    mx = re.search(r"max_volume: (-?[\d.]+) dB", out)
    h0 = re.search(r"histogram_0db: (\d+)", out)
    return {"max_volume_db": float(mx.group(1)) if mx else None,
            "clipped_samples": int(h0.group(1)) if h0 else 0}


def inspect_technical(video_path: Path, audio_path: Path | None) -> dict:
    """블랙/무음/정지/클리핑을 4종 병렬 검출. 무음·클리핑은 오디오(없으면 영상)에서."""
    media = audio_path or video_path
    with ThreadPoolExecutor(max_workers=4) as pool:
        fb = pool.submit(_detect_black, video_path)
        fs = pool.submit(_detect_silence, media)
        ff = pool.submit(_detect_freeze, video_path)
        fc = pool.submit(_detect_clipping, media)
        return {"black": fb.result(), "silence": fs.result(), "freeze": ff.result(), "clipping": fc.result()}


# ─────────────────────────────────────────────────────────────
# 오케스트레이터: ①~⑦을 순차 실행. 이미 존재하는 산출물(provided)은 재사용하고 없는 단계만 수행한다.
STEP_NAMES = ["대기", "오디오 추출", "Whisper 자막", "GPT 자막 교정", "장면 감지",
              "장면별 GPT 분석", "메타데이터 통합", "기술 검토"]


def _artifact(job_dir: Path, base: str, step: int, label: str, ext: str, ts: str) -> Path:
    """산출물 파일명 규칙: (파일명)_(진행단계)_(산출물명)_(시간).ext"""
    return job_dir / f"{base}_{step}_{label}_{ts}{ext}"


def _write_manifest(job_dir: Path, base: str, ts: str, status: str, artifacts: dict, scene_count: int) -> None:
    manifest = {"base": base, "timestamp": ts, "status": status, "artifacts": artifacts, "scene_count": scene_count}
    (job_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(job_id: str, jobs: dict, job_dir: Path, base: str, ts: str,
                 video_path: Path | None, provided: dict, options: dict | None = None) -> None:
    """provided: {역할: 경로}. 역할 ∈ {audio, original, corrected, metadata, tech}. 있는 산출물은 재사용.
    options: {whisper_backend, whisper_model, llm_model}."""
    options = options or {}
    whisper_backend = options.get("whisper_backend", "openai")
    whisper_model = options.get("whisper_model", "base")
    llm_model = options.get("llm_model", TEXT_MODEL)

    job = jobs[job_id]
    artifacts = {role: p.name for role, p in provided.items()}
    audio_path = provided.get("audio")
    try:
        job["status"] = "running"

        # ②③ 자막: corrected 가 있으면 ①②③ 모두 건너뜀
        if "corrected" in provided:
            job["step"], job["step_name"] = 3, STEP_NAMES[3]
            corrected_srt = provided["corrected"].read_text(encoding="utf-8")
            job["warnings"] = []
        else:
            # original 이 없으면 ①오디오 → ②Whisper 로 생성
            if "original" in provided:
                original_srt = provided["original"].read_text(encoding="utf-8")
            else:
                # ① 오디오 추출 (제공된 audio 재사용, 없으면 영상에서 추출)
                job["step"], job["step_name"] = 1, STEP_NAMES[1]
                if audio_path is None:
                    if not video_path:
                        raise RuntimeError("자막 생성을 위해 영상 또는 오디오(audio) 산출물이 필요합니다.")
                    audio_path = _artifact(job_dir, base, 1, "audio", ".mp3", ts)
                    extract_audio(video_path, audio_path)
                artifacts["audio"] = audio_path.name
                # ② Whisper 자막 (OpenAI API 또는 로컬 faster-whisper)
                job["step"], job["step_name"] = 2, STEP_NAMES[2]
                if whisper_backend == "local":
                    original_srt = transcribe_local(audio_path, whisper_model)
                else:
                    original_srt = transcribe(split_audio(audio_path))
                original_path = _artifact(job_dir, base, 2, "original", ".srt", ts)
                original_path.write_text(original_srt, encoding="utf-8")
                artifacts["original"] = original_path.name
            if "original" in provided:
                artifacts["original"] = provided["original"].name
            # ③ GPT 자막 교정
            job["step"], job["step_name"] = 3, STEP_NAMES[3]
            corrected_srt, warnings = correct_subtitles(original_srt, model=llm_model)
            corrected_path = _artifact(job_dir, base, 3, "corrected", ".srt", ts)
            corrected_path.write_text(corrected_srt, encoding="utf-8")
            job["warnings"] = warnings
            artifacts["corrected"] = corrected_path.name
        corrected_subs = srt_utils.parse(corrected_srt)

        # ④⑤⑥ 장면 분석: metadata 가 있으면 그대로 사용(보기 전용), 없으면 영상에서 생성
        if "metadata" in provided:
            job["step"], job["step_name"] = 6, STEP_NAMES[6]
            metadata = json.loads(provided["metadata"].read_text(encoding="utf-8"))
            artifacts["metadata"] = provided["metadata"].name
        else:
            if not video_path:
                raise RuntimeError("④ 장면 분석에는 원본 영상이 필요합니다. 영상을 함께 올려주세요.")
            # ④ 장면 감지
            job["step"], job["step_name"] = 4, STEP_NAMES[4]
            scenes = detect_scenes(video_path)
            # ⑤ 장면별 프레임 + GPT 분석 (병렬, 진행 카운터)
            job["step"], job["step_name"] = 5, STEP_NAMES[5]
            (job_dir / "frames").mkdir(exist_ok=True)
            job["scene_total"], job["scene_done"] = len(scenes), 0
            lock = threading.Lock()

            def process_scene(i: int, start: float, end: float) -> dict:
                frame_rel = f"frames/{base}_5_frame{i:03d}_{ts}.jpg"
                extract_frame(video_path, (start + end) / 2, job_dir / frame_rel)
                subs_in_scene = [s.content for s in srt_utils.slice_by_range(corrected_subs, start, end)]
                analysis = analyze_scene(job_dir / frame_rel, "\n".join(subs_in_scene), model=llm_model)
                with lock:
                    job["scene_done"] += 1
                return {
                    "index": i,
                    "start": _fmt(start), "end": _fmt(end),
                    "start_sec": round(start, 3), "end_sec": round(end, 3),
                    "frame": frame_rel,
                    "subtitles": subs_in_scene,
                    "analysis": analysis,
                }

            with ThreadPoolExecutor(max_workers=SCENE_WORKERS) as pool:
                futures = [pool.submit(process_scene, i, s, e) for i, (s, e) in enumerate(scenes, start=1)]
                scene_entries = [f.result() for f in futures]

            # ⑥ 메타데이터 통합
            job["step"], job["step_name"] = 6, STEP_NAMES[6]
            metadata = {
                "video": video_path.name,
                "duration_sec": round(probe_duration(video_path), 3),
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "scene_count": len(scene_entries),
                "scenes": scene_entries,
            }
            metadata_path = _artifact(job_dir, base, 6, "metadata", ".json", ts)
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts["metadata"] = metadata_path.name

        # ⑦ 기술 검토 (제공되면 재사용, 없고 영상이 있으면 검출)
        if "tech" in provided:
            artifacts["tech"] = provided["tech"].name
        elif video_path:
            job["step"], job["step_name"] = 7, STEP_NAMES[7]
            tech = inspect_technical(video_path, audio_path)
            tech_path = _artifact(job_dir, base, 7, "tech", ".json", ts)
            tech_path.write_text(json.dumps(tech, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts["tech"] = tech_path.name

        job["artifacts"] = artifacts
        _write_manifest(job_dir, base, ts, "done", artifacts, metadata.get("scene_count", 0))
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        _write_manifest(job_dir, base, ts, "error", artifacts, 0)

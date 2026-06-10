# 영상 자동 분석 파이프라인 — 구현 계획

## 개요

영상 파일 1개를 업로드하면 ①오디오 추출 → ②Whisper 자막 → ③GPT 자막 교정 → ④장면 감지 → ⑤장면별 GPT 분석 → ⑥메타데이터 통합을 거쳐 **원본 SRT, 교정 SRT, 장면 메타데이터 JSON**을 내려받는 웹 페이지.

**확정된 결정:**
- Whisper: **OpenAI Whisper API** (`whisper-1` — SRT 직접 반환 지원)
- 장면 분석(⑤): **대표 프레임 이미지 + 해당 구간 자막**을 GPT 비전 모델로 분석
- 스택: **Python FastAPI + 단일 HTML 페이지** (바닐라 JS, 2초 폴링)

**핵심 제약:**
- GPT 교정 시 **타임코드 절대 수정 금지** (③은 텍스트만 교정)
- API 키는 `.env` (`OPENAI_API_KEY` 하나)

## 파일 구조

```
claude_md\
├── app.py            FastAPI: 엔드포인트, 잡 레지스트리, BackgroundTasks (~120줄)
├── pipeline.py       6단계 파이프라인 + 오케스트레이터 (~250줄)
├── srt_utils.py      SRT 파싱/조립/시프트/병합/검증 (~80줄)
├── static\index.html 업로드 → 진행상황 폴링 → 다운로드 링크 (~120줄)
├── requirements.txt  fastapi, uvicorn[standard], python-multipart, python-dotenv, openai, srt, scenedetect[opencv]
├── .env.example      OPENAI_API_KEY=
├── .gitignore        .env, jobs/, __pycache__/
└── jobs\<job_id>\    런타임 산출물: input.mp4, audio.mp3, original.srt, corrected.srt, frames\, metadata.json
```

외부 의존: **ffmpeg** (PATH 필요 — 서버 시작 시 `shutil.which("ffmpeg")`로 확인, 없으면 명확한 에러로 종료. 설치: `winget install Gyan.FFmpeg`)

## 파이프라인 단계별 설계

### ① 오디오 추출
```
ffmpeg -y -i input.mp4 -vn -ac 1 -ar 16000 -b:a 64k audio.mp3
```
- 64kbps mono ≈ 시간당 28.8MB → 약 50분 이하 영상은 Whisper API 25MB 제한 통과.
- 24MB(안전 마진) 초과 시: 시간 기준으로 분할(`-c copy`), 각 청크의 시작 오프셋(초)을 기록.

### ② Whisper 자막
- `client.audio.transcriptions.create(model="whisper-1", file=..., response_format="srt")` — 청크별 호출.
- `whisper-1`을 쓰는 이유: SRT 포맷을 직접 반환 (gpt-4o-transcribe는 json/text만 지원 → 타임코드 재구성 필요해짐).
- 다중 청크: 각 SRT를 파싱 → 오프셋만큼 시프트 → 병합·재번호 → `original.srt` 저장.

### ③ GPT 자막 교정 — 타임코드 안전 장치
**타임코드는 서버 밖으로 나가지 않는다. 구조적으로 수정 불가능하게 만들고, 최종 검증으로 이중 확인한다.**

1. `original.srt`를 큐 객체 `(index, start, end, text)`로 파싱.
2. GPT에는 **텍스트만** 전송: `{"lines": ["...", ...]}` JSON, 50개 단위 배치.
3. 프롬프트: "오탈자·맞춤법·띄어쓰기만 교정. 번역·의역·병합·분할·재배열·삭제 금지. 정확히 N개를 같은 순서의 JSON으로 반환." (`response_format: json_object`)
4. 검증: 반환 개수 == 전송 개수. 불일치 → 1회 재시도 → 그래도 실패 시 해당 배치는 원문 유지(잡 상태에 경고 기록). 전체 잡을 실패시키지 않음.
5. 교정 텍스트를 **원본 큐 객체**에 써넣어 `corrected.srt` 재조립 — start/end 필드는 손대지 않음.
6. 최종 assert: 원본/교정본의 타임코드 라인(`-->` 라인) 완전 일치. 불일치 시 에러.

### ④ 장면 감지
- **PySceneDetect `ContentDetector`** (기본 threshold 27): `detect(video_path, ContentDetector())` 한 줄로 장면 구간 `(start, end)` 리스트를 구조화해 반환. (ffmpeg scene 필터는 컷 프레임만 출력 + stderr 파싱 필요 → 부적합)
- 컷 0개 감지 시 → 전체 영상을 1개 장면으로 처리.

### ⑤ 장면별 GPT 분석
- 대표 프레임 = 장면 중간 지점: `ffmpeg -ss <mid> -i input.mp4 -frames:v 1 -q:v 3 frames/scene_NNN.jpg`
- 장면 자막 = **교정된** 자막에서 구간 겹침으로 추출 (`cue.start < scene.end and cue.end > scene.start`).
- 프레임(base64 JPEG) + 장면 자막 → GPT 비전 모델, JSON 응답 요구. 순차 호출.
- 분석 필드: `summary`, `visual_description`, `keywords`.

### ⑥ 메타데이터 통합 — `metadata.json` 스키마
```json
{
  "video": "input.mp4",
  "duration_sec": 312.5,
  "created_at": "2026-06-10T12:00:00+09:00",
  "scene_count": 14,
  "scenes": [
    {
      "index": 1,
      "start": "00:00:00,000",
      "end": "00:00:12,400",
      "start_sec": 0.0,
      "end_sec": 12.4,
      "frame": "frames/scene_001.jpg",
      "subtitles": ["자막 1", "자막 2"],
      "analysis": {
        "summary": "...",
        "visual_description": "...",
        "keywords": ["...", "..."]
      }
    }
  ]
}
```

## 실행 모델

- 업로드당 FastAPI `BackgroundTasks` 1개 — Celery/Redis 불필요 (개인용 단일 사용자 도구).
- 잡 상태: 인메모리 dict (`uuid4().hex` 키) — `status(queued|running|done|error)`, `step(0–6)`, `step_name`, `error`, `files`. 재시작 시 소실되나 산출물은 `jobs/<id>/`에 디스크 보존.
- 순차 실행 ①→⑥: ⑤가 ③(교정 자막)과 ④(장면 구간) 모두에 의존하고, 전체 시간은 OpenAI 호출이 지배하므로 병렬화 이득 없음.
- 진행상황: 프론트에서 2초 폴링 — 6단계 거친 진행 표시에 SSE는 과함.
- 모델명은 `pipeline.py` 상수: `TRANSCRIBE_MODEL="whisper-1"`, `TEXT_MODEL="gpt-4o-mini"`, `VISION_MODEL="gpt-4o-mini"`.

## 엔드포인트

| Method | Path | 동작 |
|---|---|---|
| GET | `/` | `static/index.html` 서빙 |
| POST | `/upload` | 영상 저장 → 잡 등록 → 백그라운드 실행 → `{"job_id": "..."}` |
| GET | `/jobs/{id}` | `{status, step, step_name, error, files}` (완료 시 files 포함), 미존재 시 404 |
| GET | `/jobs/{id}/files/{name}` | 다운로드 — `original.srt` / `corrected.srt` / `metadata.json` 정확한 이름 화이트리스트 (경로 탐색 차단) |

## 구현 순서 및 단계별 검증

1. **스켈레톤**: requirements.txt, .env.example, .gitignore, app.py(기동 체크 + 정적 페이지)
   → 검증: `pip install` 성공, 서버 기동, `.env`/ffmpeg 누락 시 명확한 에러
2. **srt_utils.py**: parse / compose / shift / merge / slice / timecode_lines
   → 검증: 3큐 샘플로 라운드트립 타임코드 보존, 10초 shift 동작
3. **①②** (pipeline.py)
   → 검증: 샘플 영상으로 audio.mp3·original.srt 생성, `AUDIO_LIMIT_BYTES` 낮춰 분할 경로 테스트 — 병합 타임코드 단조 증가 확인
4. **③** 교정
   → 검증: 원본/교정 SRT의 `-->` 라인 diff **바이트 단위 동일**, 개수 불일치 모의 응답으로 폴백(원문 유지) 확인
5. **④⑤⑥**
   → 검증: 프레임 수 == 장면 수, metadata.json 구간이 0→duration을 빈틈/중복 없이 커버, 자막이 올바른 장면에 배치
6. **app.py 연결** (잡 레지스트리, 백그라운드, 상태/다운로드)
   → 검증: curl 업로드 → 폴링으로 step 1→6 진행 → 3개 파일 다운로드
7. **index.html**
   → 검증: 브라우저 전체 플로우 (업로드 → 단계 표시 → 다운로드 링크 3개)
8. **E2E 인수**: 음성 + 컷 2~3개 포함 30–60초 샘플 클립으로 전체 확인. 음성 없는 영상 업로드 시 에러 메시지로 종료(행 걸림 없음) 확인.
   ※ 샘플 클립은 사용자 제공 필요 (합성 테스트 패턴은 음성이 없어 부적합).

## 가정 (명시)

- 한국어/영어 음성 모두 whisper-1로 처리, 교정 프롬프트에 "원문 언어 유지" 명시.
- 동시 업로드 1개 가정, 인증·잡 정리(TTL) 없음, localhost 전용.
- 프레임 이미지는 다운로드 미제공 (metadata.json 내 경로 참조만) — 필요 시 추후 추가.

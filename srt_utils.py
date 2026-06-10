"""SRT 파싱/조립 유틸. 타임코드는 이 모듈 안에서만 다루고 외부(GPT)로 보내지 않는다."""
from datetime import timedelta

import srt


def parse(text: str) -> list[srt.Subtitle]:
    return list(srt.parse(text))


def compose(subs: list[srt.Subtitle]) -> str:
    return srt.compose(subs)


def shift(subs: list[srt.Subtitle], offset_seconds: float) -> list[srt.Subtitle]:
    delta = timedelta(seconds=offset_seconds)
    return [
        srt.Subtitle(index=s.index, start=s.start + delta, end=s.end + delta, content=s.content)
        for s in subs
    ]


def merge_and_renumber(sub_lists: list[list[srt.Subtitle]]) -> list[srt.Subtitle]:
    merged = [s for subs in sub_lists for s in subs]
    merged.sort(key=lambda s: s.start)
    return [
        srt.Subtitle(index=i, start=s.start, end=s.end, content=s.content)
        for i, s in enumerate(merged, start=1)
    ]


def slice_by_range(subs: list[srt.Subtitle], start_sec: float, end_sec: float) -> list[srt.Subtitle]:
    """구간 [start_sec, end_sec)와 겹치는 자막 큐를 반환."""
    start = timedelta(seconds=start_sec)
    end = timedelta(seconds=end_sec)
    return [s for s in subs if s.start < end and s.end > start]


def timecode_lines(text: str) -> list[str]:
    """타임코드 라인만 추출 — 교정 전후 동일성 검증용."""
    return [line for line in text.splitlines() if "-->" in line]

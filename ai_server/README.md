# ai_server — 자세 판정 FastAPI (OK / NG)

맨몸운동 영상을 받아 **사람별·반복별 자세 판정(OK / NG)과 틀린 이유**를 JSON으로 돌려줍니다.
DB 저장·회원 관리는 하지 않습니다 (Django `backend/` 담당).

## 실행

```bash
cd ai_server
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
- 상태: http://localhost:8001/health
- 판정 기준 확인: http://localhost:8001/exercises
- 테스트 화면: http://localhost:8001/docs
- 로직 테스트: `python -m pytest tests -q`

## 처리 흐름

```
영상 → YOLO Pose + 추적 (pose_pipeline.py)
     → 사람별 관절 각도 (angle_utils.py)
     → 반복(rep) 나누기 + 규칙 검사 (judges/)
     → 반복별 OK/NG + 이유 → 영상 전체 판정
```

| 파일 | 역할 |
|---|---|
| `main.py` | 앱 생성, `/health`, 시작 시 Pose 모델 로드 |
| `routers/predictions.py` | `POST /predict/video`, `GET /exercises`, 입력 검사·오류 응답 |
| `services/pose_pipeline.py` | 영상 → 사람(Track ID)별 관절 17개 |
| `services/angle_utils.py` | 관절 각도 계산, 흔들림 보정 |
| `services/judges/base.py` | 반복 감지 + 공통 판정 흐름 |
| `services/judges/squat.py`, `pushup.py` | 운동별 규칙 |
| `settings.py` | **판정 기준 숫자 모음 (미정 사항은 여기서 수정)** |

## API

`POST /predict/video` (multipart)
- `file` (필수): 영상 — mp4, avi, mov, mkv, webm / 최대 100MB
- `exercise` (선택): `squat` | `pushup` — 비우면 `DEFAULT_EXERCISE`

응답 예:
```json
{
  "status": "OK",
  "message": "1명의 스쿼트 자세를 판정했습니다.",
  "exercise": {"key": "squat", "name": "스쿼트"},
  "results": [{
    "track_id": 1, "frame_count": 240,
    "verdict": "NG", "rep_count": 3, "ok_count": 2, "ng_count": 1, "ok_ratio": 0.667,
    "reasons": ["충분히 앉지 않았어요 (무릎 최소 115°, 기준 100° 이하)"],
    "reps": [
      {"index": 1, "verdict": "OK", "reasons": [], "start_frame": 12, "end_frame": 58,
       "measures": {"knee_min_angle": 82.0, "hip_angle_at_bottom": 74.5}}
    ]
  }],
  "processing_time_ms": 5320,
  "judge": {"type": "rule", "version": "0.1"}
}
```

| 상황 | HTTP | 응답 |
|---|---|---|
| 판정 성공 | 200 | `status: "OK"` |
| 사람 없음 | 200 | `status: "NO_PERSON"`, `results: []` |
| 반복 동작 못 찾음 | 200 | `status: "NO_REP"` |
| 영상 손상 | 400 | `code: "VIDEO_UNREADABLE"` |
| 용량 초과 | 413 | `code: "FILE_TOO_LARGE"` |
| 형식 오류 | 415 | `code: "UNSUPPORTED_FORMAT"` |
| 모르는 운동 | 422 | `code: "UNKNOWN_EXERCISE"` |

## 미정 사항 → 정해지면 바꿀 곳

| 미정 사항 | 현재 임시값 | 바꿀 곳 |
|---|---|---|
| 판정할 운동 | 스쿼트, 푸시업 | `services/judges/`에 파일 추가 + `__init__.py`에 등록 |
| OK/NG 기준 각도 | 스쿼트 무릎 ≤100°, 푸시업 팔꿈치 ≤90° 등 | `settings.py` 또는 `.env` |
| 판정 단위 | 반복별 + 영상 전체 둘 다 반환 | 응답에서 필요한 것만 사용 |
| 영상 전체 OK 기준 | 반복 중 OK 70% 이상 | `OVERALL_OK_RATIO` |
| 운동 선택 방식 | 요청의 `exercise` (없으면 기본값) | `DEFAULT_EXERCISE` |
| 판정 방식 | 관절 각도 규칙 | 학습 모델로 바꾸면 judge 클래스만 새로 작성 |

권장 촬영: **옆모습, 전신이 보이게, 2회 이상 반복.**

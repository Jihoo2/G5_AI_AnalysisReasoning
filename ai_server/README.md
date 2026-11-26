# ai_server — FastAPI AI 서버

영상을 받아 **`OK` 또는 `NG`**를 돌려줍니다.
- 정답 동작은 **파일 이름**에서 읽어요: `S01_A001_T01.mp4` → `A001` → `bend_return`
- AI가 인식한 동작 = 정답 → `OK`, 다르면 `NG` (사람을 못 찾는 등 판단 불가도 `NG`)

DB 저장·회원·화면은 하지 않습니다 (Django `backend/` 담당).

## 구조

```
ai_server/
├── main.py            서버 전부 — 위에서 아래로 처리 순서대로
│                        1 설정 → 2 Feature → 3 모델 로드 → 4 Pose → 5 판정 → 6 API
├── tests/test_main.py 테스트
├── requirements.txt
├── project.json       팀장 설정 (Pose 모델·신뢰도)
├── data/              테스트 영상 (Git 제외)
└── models/
    ├── rf_action_baseline.joblib   ← 팀장에게 받기 (Git 제외)
    ├── yolo11n-pose.pt             ← 팀장에게 받기 (Git 제외)
    ├── rf_feature_columns.json
    └── rf_training_info.json
```

## 실행

```bash
cd ai_server
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest tests -q           # 8 passed
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
- 상태: http://localhost:8001/health (`rf_model_ready`, `pose_model_ready` 둘 다 true)
- 테스트 화면: http://localhost:8001/docs

## API

| API | 하는 일 |
|---|---|
| `POST /predict/video` | 영상 업로드(`file`) → `OK` / `NG`. `expected_action`을 주면 파일 이름 대신 그 값을 정답으로 |
| `GET /predict/file/{파일명}` | `data/` 폴더 영상을 이름으로 → `OK` / `NG`. 예: `/predict/file/S01_A001_T01.mp4` |
| `GET /data` | `data/` 폴더의 영상 목록 |
| `GET /health` | 서버·모델 준비 상태 |
| `GET /actions` | 인식 가능한 동작 목록 |

`?detail=true`를 붙이면 `OK`/`NG` 대신 상세 JSON을 돌려줘요 (인식 동작, 신뢰도 등 확인용).

| 파일 이름 ID | 정답 동작 |
|---|---|
| A001 | `bend_return` 허리 굽혔다 펴기 |
| A002 | `leg_raise_lower` 한쪽 다리 들었다 내리기 |
| A003 | `walk_turn_walk` 걷다가 방향 바꾸기 |
| A004 | `drink_return` 음료 마시기 |
| A005 | `sit_stand` 앉았다 일어나기 |
| A006 | `wave` 손 흔들기 |

| 상황 | HTTP | 응답 |
|---|---|---|
| 판정 성공 | 200 | `OK` 또는 `NG` |
| 사람 없음·너무 짧음 | 200 | `NG` |
| 영상 손상 | 400 | `VIDEO_UNREADABLE` |
| data 폴더에 파일 없음 | 404 | `FILE_NOT_FOUND` |
| 용량 초과 (100MB) | 413 | `FILE_TOO_LARGE` |
| 형식 오류 | 415 | `UNSUPPORTED_FORMAT` |
| 파일 이름에 A001~A006 없음 | 422 | `NO_LABEL_IN_FILENAME` |
| 모델 없음 | 503 | `MODEL_NOT_READY` |

오류 응답 모양: `{"detail": {"code": "VIDEO_UNREADABLE", "message": "..."}}`

## Django에서 호출하기 (`views.py`)

```python
import requests

AI_URL = "http://localhost:8001/predict/video"

with open(record.video_file.path, "rb") as f:
    res = requests.post(AI_URL, files={"file": (record.video_file.name.split("/")[-1], f)}, timeout=300)

if res.status_code == 200:
    record.status, record.verdict = "DONE", res.text.strip()   # "OK" 또는 "NG"
else:
    record.status = "FAILED"                                    # res.json()["detail"]["message"] 에 이유
record.save()
```
보내는 파일 이름에 `A001`~`A006`이 들어 있어야 해요 (정답을 거기서 읽음).

## 판정 흐름

1. YOLO11 Pose(`project.json`) + bytetrack으로 사람별 관절 17개 — 프레임 2장에 1장 분석
2. 프레임마다 38개: 정규화 좌표 34 + 팔꿈치·무릎 각도 4 (`rf_feature_columns.json`과 순서 일치를 테스트로 확인)
3. 5초 구간을 0.25초씩 밀며 30 Step으로 보간 → 1,140개
4. Random Forest 확률을 구간마다 구해 평균 → 인식 동작
5. 파일 이름의 정답과 같으면 **OK**, 다르면 **NG**

검증: 학습 영상 `S01_A001_T01.mp4` → `OK`

## 참고

| 항목 | 현재 | 바꿀 곳 (`main.py`) |
|---|---|---|
| 좌표 정규화 | 엉덩이 중심 기준, 몸통 길이로 나눔 | `normalize_keypoints()` |
| 판정 기준값 | 5초 구간, 0.25초 간격, 신뢰도 0.55 | "1. 설정" 구역 |
| OK/NG 규칙 | 정답 = 인식 동작이면 OK | `judge()` |
| 분석 속도 | 프레임 2장에 1장 | 환경변수 `FRAME_STRIDE` |

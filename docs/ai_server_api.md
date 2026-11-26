# ai_server — FastAPI AI 서버

영상을 받아 **사람별 동작 인식(6개 중 하나) + 신뢰도 + OK/NG**를 JSON으로 돌려줍니다.
DB 저장·회원·화면은 하지 않습니다 (Django `backend/` 담당).

## 구조

```
ai_server/
├── main.py              서버 전부 — 위에서 아래로 처리 순서대로
│                          1 설정 → 2 Feature → 3 모델 로드 → 4 Pose → 5 판정 → 6 API
├── client/ai_client.py  백엔드(Celery)가 복사해서 쓰는 호출 함수
├── tests/test_main.py   테스트
├── requirements.txt · Dockerfile
├── project.json         팀장 설정 (Pose 모델·신뢰도)
├── configs/             팀장 설정 (day07 Window·신뢰도, day08 추적기)
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
python -m pytest tests -q           # 11 passed
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
- 상태: http://localhost:8001/health (`rf_model_ready`, `pose_model_ready` 둘 다 true)
- 테스트 화면: http://localhost:8001/docs

## API

**응답은 기본적으로 `OK` 또는 `NG` 두 글자만** 돌려줍니다.
- 정답 동작은 **파일 이름**에서 읽어요: `S01_A001_T01.mp4` → `A001` → `bend_return`
- AI가 인식한 동작 = 정답 → `OK`, 다르면 `NG` (사람을 못 찾는 등 판단 불가도 `NG`)
- `?detail=true`를 붙이면 아래 상세 JSON을 돌려줘요 (확인용·백엔드용)

| 파일 이름 ID | 정답 동작 |
|---|---|
| A001 | `bend_return` 허리 굽혔다 펴기 |
| A002 | `leg_raise_lower` 한쪽 다리 들었다 내리기 |
| A003 | `walk_turn_walk` 걷다가 방향 바꾸기 |
| A004 | `drink_return` 음료 마시기 |
| A005 | `sit_stand` 앉았다 일어나기 |
| A006 | `wave` 손 흔들기 |

| API | 하는 일 |
|---|---|
| `POST /predict/video` | 영상 업로드 → `OK`/`NG` (`expected_action`을 주면 파일 이름 대신 그 값을 정답으로) |
| `GET /data` | `ai_server/data/` 폴더의 영상 목록 |
| `GET /predict/file/{파일명}` | `data/` 폴더 영상을 이름으로 분석 → `OK`/`NG`. 예: `/predict/file/S01_A001_T01.mp4` |

상세 JSON (`?detail=true`):

```json
{
  "verdict": "OK",
  "status": "OK",
  "message": "1명의 동작을 분석했습니다.",
  "expected_action": "wave",
  "summary": {
    "track_id": 1, "people_count": 1,
    "action_label": "wave", "action_name": "손 흔들기",
    "confidence": 0.81, "confidence_percent": 81, "is_uncertain": false,
    "verdict": "OK",
    "display_text": "OK · 손 흔들기 (81%) · '손 흔들기' 동작을 신뢰도 81%로 인식했습니다."
  },
  "results": [{
    "track_id": 1, "action_label": "wave", "action_name": "손 흔들기",
    "confidence": 0.81, "is_uncertain": false, "verdict": "OK", "reason": "...",
    "window_count": 12, "frame_count": 120,
    "votes": {"wave": 10}, "probabilities": {"wave": 0.81, "...": 0.0}
  }],
  "processing_time_ms": 1350
}
```
- **`summary`**: 가장 오래 보인 사람 1명. 백엔드는 이것만 저장·표시해도 됩니다. `display_text`는 화면에 그대로 출력.
- **`results`**: 사람별 상세 (여러 명일 때 `PersonPrediction` 저장용)

| 상황 | HTTP | 응답 |
|---|---|---|
| 분석 성공 | 200 | `status: "OK"` |
| 사람 없음 | 200 | `status: "NO_PERSON"` |
| 너무 짧거나 가려짐 | 200 | `status: "TOO_SHORT"` |
| 영상 손상 | 400 | `code: "VIDEO_UNREADABLE"` |
| 용량 초과 | 413 | `FILE_TOO_LARGE` |
| 형식 오류 | 415 | `UNSUPPORTED_FORMAT` |
| 모르는 동작 | 422 | `UNKNOWN_ACTION` |
| 파일 이름에 A001~A006 없음 | 422 | `NO_LABEL_IN_FILENAME` |
| data 폴더에 파일 없음 | 404 | `FILE_NOT_FOUND` |
| 모델 없음 | 503 | `MODEL_NOT_READY` |

`GET /actions`: 인식 가능한 동작(영문·한글)과 판정 기준

## 백엔드 연결 — `client/ai_client.py`

`backend/`의 앱 폴더에 복사하면 함수 하나로 끝납니다.

```python
from .ai_client import analyze_video

result = analyze_video(job.video_file.path, expected_action=job.expected_action)
if result["ok"]:                       # 사람이 없어도 '분석 완료'
    job.status = "COMPLETED"
    job.result_text = result["summary"]["display_text"]
    for p in result["people"]:
        PersonPrediction.objects.create(job=job, **p)
else:                                  # 연결 실패·시간 초과·영상 손상 등
    job.status = "FAILED"
    job.error_message = result["error_message"]
job.save()
```
`people` 필드: `track_id, action_label, action_name, confidence, is_uncertain, verdict, reason, window_count, frame_count`
주소: 환경변수 `AI_SERVICE_URL` (Docker: `http://fastapi:8001`)

## 판정 흐름

1. YOLO11 Pose(`project.json`) + bytetrack(`day08`)로 사람별 관절 17개 — 프레임 2장에 1장 분석(`FRAME_STRIDE`)
2. 프레임마다 38개: 정규화 좌표 34 + 팔꿈치·무릎 각도 4 (`rf_feature_columns.json`과 순서 일치를 테스트로 확인)
3. 5초 Window를 0.25초씩 밀며(`day07`) 30 Step으로 보간 → 1,140개
4. Random Forest 확률을 Window마다 구해 평균 → 동작 + 신뢰도 (0.55 미만은 불확실)
5. 파일 이름의 정답(또는 `expected_action`)과 같으면 **OK**, 다르면 **NG**

## ⚠️ 팀 확인 필요 (모두 main.py 안)

| 항목 | 현재 가정 | 바꿀 곳 |
|---|---|---|
| 좌표 정규화 | 엉덩이 중심 기준, 몸통 길이로 나눔 | `normalize_keypoints()` — 학습 코드 `feature_utils.py`와 같게 |
| 각도 단위 | 도(0~180) | `frame_features()` |
| 가려진 관절 값 | 0으로 채움 | `flatten()` |
| OK/NG 기준 | 정답(파일 이름) = 인식 동작이면 OK (신뢰도는 따지지 않음) | `judge()` |
| scikit-learn 버전 | 1.4 이상 | `requirements.txt` — 팀장 학습 버전과 맞추기 |

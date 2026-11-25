# ai_server — FastAPI AI 서버

영상을 받아 **사람별 동작 인식(6개 중 하나) + 신뢰도 + OK/NG**를 JSON으로 돌려줍니다.
DB 저장·회원·화면은 하지 않습니다 (Django `backend/` 담당).

## 준비

모델 파일 2개는 용량 때문에 Git에 없습니다. 팀장에게 받아 `models/`에 넣으세요.

```
ai_server/
├── models/
│   ├── rf_action_baseline.joblib   ← 팀장에게 받기 (Git 제외)
│   ├── yolo11n-pose.pt             ← 팀장에게 받기 (Git 제외)
│   ├── rf_feature_columns.json     (Git 포함)
│   └── rf_training_info.json       (Git 포함)
├── configs/                        팀장 설정 (day07 Window·신뢰도, day08 추적기)
├── project.json                    팀장 설정 (Pose 모델·신뢰도)
├── app_config.py                   위 설정 파일을 읽어 한곳에 모음
├── main.py                         앱 시작, 모델 로드, /health
├── routers/predictions.py          /predict/video, /actions
└── services/
    ├── features.py                 관절 → 프레임당 38개 → 30 Step → 1,140개
    ├── pose_pipeline.py            영상 → YOLO Pose + 추적 → 사람별 Feature
    ├── model_loader.py             Random Forest + json 로드·검사
    └── inference.py                5초 Window 예측·평균 → 동작 + OK/NG
```

## 실행

```bash
cd ai_server
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest tests -q           # 6 passed
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```
- 상태: http://localhost:8001/health (`rf_model_ready`, `pose_model_ready` 둘 다 true)
- 테스트 화면: http://localhost:8001/docs

## API

`POST /predict/video` (multipart/form-data)
- `file` (필수): mp4·avi·mov·mkv·webm, 최대 100MB
- `expected_action` (선택): 하려던 동작 — 주면 OK/NG 판정

```json
{
  "status": "OK",
  "message": "1명의 동작을 분석했습니다.",
  "expected_action": "wave",
  "results": [{
    "track_id": 1,
    "action_label": "wave", "confidence": 0.81, "is_uncertain": false,
    "verdict": "OK", "reason": "wave 동작을 신뢰도 81%로 인식했습니다.",
    "window_count": 12, "frame_count": 240,
    "votes": {"wave": 10, "drink_return": 1},
    "probabilities": {"wave": 0.81, "drink_return": 0.08, "...": 0.0}
  }],
  "processing_time_ms": 4210
}
```

| 상황 | HTTP | 응답 |
|---|---|---|
| 분석 성공 | 200 | `status: "OK"` |
| 사람 없음 | 200 | `status: "NO_PERSON"`, `results: []` |
| 너무 짧거나 가려짐 | 200 | `status: "TOO_SHORT"` |
| 영상 손상 | 400 | `code: "VIDEO_UNREADABLE"` |
| 용량 초과 | 413 | `FILE_TOO_LARGE` |
| 형식 오류 | 415 | `UNSUPPORTED_FORMAT` |
| 모르는 동작 | 422 | `UNKNOWN_ACTION` |
| 모델 없음 | 503 | `MODEL_NOT_READY` |

`GET /actions`: 인식 가능한 동작 목록과 판정 기준

## 판정 흐름

1. YOLO11 Pose(`project.json`)로 사람별 관절 17개 + Track ID(`day08`의 bytetrack)
2. 프레임마다 38개: 정규화 좌표 34 + 팔꿈치·무릎 각도 4 (`rf_feature_columns.json` 순서와 테스트로 일치 확인)
3. 5초 Window를 0.25초씩 밀며(`day07`) 30 Step으로 보간 → 1,140개
4. Random Forest 확률을 Window마다 구해 평균 → 동작 + 신뢰도
5. 신뢰도 0.55 미만이면 `is_uncertain`
6. `expected_action`과 같고 불확실하지 않으면 **OK**, 아니면 **NG**

## ⚠️ 팀 확인 필요

| 항목 | 현재 가정 | 바꿀 곳 |
|---|---|---|
| 좌표 정규화 방법 | 엉덩이 중심 기준, 몸통 길이로 나눔 | `services/features.py`의 `normalize_keypoints()` — 학습 코드 `feature_utils.py`와 같게 |
| 각도 단위 | 도(0~180) | `services/features.py`의 `frame_features()` |
| 가려진 관절 값 | 0으로 채움 | `services/features.py`의 `flatten()` |
| OK/NG 기준 | 하려던 동작 = 인식 동작, 신뢰도 ≥ 0.55 | `services/inference.py`의 `judge()` |
| scikit-learn 버전 | 1.4 이상 | `requirements.txt` — 팀장 학습 버전과 맞추기 |

# AI 분석 버튼 파이프라인 (STEP 10)

> G5_AI_AnalysisReasoning — "AI 분석 시작" 버튼 하나가 화면 → Django → FastAPI → DB → 화면으로 돌아오는 전체 흐름.
> 각자 자기 코드를 추적해 채운 표를 합친 것 (2026-11-26).
> ★ = 담당이 바뀌는 지점 (주고받는 값이 팀 약속)

## 한눈에 보기

```
[AI 분석 시작]  screen.html (유나)
 → screen.js  startInference()  FormData(video, action)  fetch POST          (태형)
 → config/urls.py  api/v1/  →  api_urls.py  analyze/  →  views.analyze()      (지후)
 → InferenceRecord.objects.create()  status=RECEIVED                          (성환)
 → requests.post  :8001/predict/video?detail=true  (file, expected_action)    (지후)
 → predict_video → analyze_file → YOLO Pose → RF → judge → OK/NG              (효근)
 → record.verdict = ...  status=DONE  save()                                   (지후)
 → _result(record)  {id, status, verdict}                                      (성환)
 → screen.js  fetch GET /api/v1/analyze/{id}/  →  views.result()               (태형 → 성환)
 → results.dataset.verdict  →  CSS 배지 강조                                   (태형 → 유나)
```

## 전체 표

| # | 단계 | 파일 | 함수 / 요소 | 넘기는 방식 | 왜? | 실패하면? | 담당 |
|---|---|---|---|---|---|---|---|
| 1 | 화면 요소 배치 | `screen.html` | `#videoFile`, `#action`, `#start`, `.results` | `index()`가 HTML을 렌더링 → JS가 앞의 세 요소는 `id`, `.results`는 `class`로 찾음 | JS가 영상·동작을 읽고, 버튼을 분석 함수에 연결하고, 결과를 표시하기 위해 | 이름이 다르면 JS가 요소를 못 찾음. `#start`가 없으면 클릭 연결에서 오류 → 분석 시작 불가 | 유나 |
| 2 | CSRF 토큰 삽입 | `screen.html` | `{% csrf_token %}` | 숨은 입력 `csrfmiddlewaretoken` → JS가 읽어 `X-CSRFToken` 헤더에 담음 | Django가 이 페이지에서 온 POST인지 검증하도록 | 요소가 없으면 JS가 `.value`를 읽는 순간 오류(이벤트 연결 전이라 요청도 안 감). 값이 틀리면 Django가 **403** | 유나 |
| 3 | 파일 선택 → 미리보기 | `screen.js` | `previewFile()` ← `videoFile.onchange` | 선택한 File의 임시 URL → `video.src`, `resetAnalysis()` | 고른 영상을 확인하고 이전 결과를 지움 | 파일이 비면 미리보기를 비우고 버튼 비활성화 | 태형 |
| 4 | 버튼 클릭 | `screen.js` | `startInference()` ← `start.onclick` | 파일을 읽고 버튼을 비활성화한 뒤 5번으로 | 연속 클릭으로 같은 영상이 중복 전송되는 걸 막음 | 파일 없음·카메라 탭·이미 요청 중이면 즉시 종료 | 태형 |
| 5 | 보낼 데이터 만들기 | `screen.js` | `startInference()` | `FormData`: `video`, `action` | Django가 파일과 비교할 동작을 한 요청에서 읽도록 | 동작을 안 고르면 `action`이 빈 문자열 → FastAPI가 파일명(A001~A006)에서 정답을 찾음 | 태형 |
| 6 ★ | Django로 전송 | `screen.js` | `fetch` POST | `data-api`(`/api/v1/analyze/`) + `X-CSRFToken` | 영상·동작을 Django에 보내 저장과 AI 분석을 맡기려고. CSRF 헤더는 Django가 POST를 받아 주게 하려고 | 네트워크·HTTP 오류면 중단하고 실패 문구 표시 | 태형 → 지후 |
| 7 | API 주소 갈래 | `config/urls.py` | `path("api/v1/", include("analysis.api_urls"))` | URL 매칭 | 화면 주소(`/analyses/`)와 API 주소(`/api/v1/`)를 나눠 관리. API가 늘어도 여기는 한 줄 | `api/v1/`로 시작하는 주소가 없으면 **404** | 지후 |
| 8 | API → 함수 | `analysis/api_urls.py` | `path("analyze/", views.analyze)` | URL 매칭 | 남은 주소 `analyze/`를 분석 함수에 연결. `@api_view(["POST"])`라 POST만 받음 | 주소가 틀리면 **404**, GET으로 오면 **405** | 지후 |
| 9 | 영상 받기 | `views.py` | `analyze()` | `request.FILES["video"]` | 파일은 글자(POST)가 아니라 FILES로 들어옴. `"video"`는 JS와 정한 약속 이름 | 영상이 없으면 **400** + `{"error": "video 파일이 없습니다."}` | 지후 |
| 10 | 업로드 기록 저장 | `views.py` | `InferenceRecord.objects.create(video_file=video)` | ORM → SQLite, 파일 → `media/videos/` (MEDIA_ROOT) | AI 호출 **전에** 기록(RECEIVED)을 남겨 AI가 실패해도 흔적과 id가 남게. 영상은 크니까 DB엔 경로만 (`media/`는 GitHub 제외) | 저장 자체가 실패하면(DB 잠김·디스크·권한) 처리 코드가 없어 **500** → 화면 실패 문구 | 성환 |
| 11 ★ | FastAPI 호출 | `views.py` | `analyze()` → `requests.post()` | `file`(원래 파일명) + `expected_action`, `?detail=true`, timeout 120 | 판정은 FastAPI가 함 — 모델과 판정 기준을 한 곳에만 두려고. `detail=true`가 없으면 JSON이 아니라 `"OK"` 글자만 옴 | FastAPI 꺼짐·시간 초과·4xx/5xx → `except` → `status = "FAILED"`, verdict 빈칸 | 지후 → 효근 |
| 12 | 요청 받기 | `ai_server/main.py` | `predict_video()` ← `@app.post("/predict/video")` | `UploadFile "file"` + `Form expected_action` | 영상은 바이너리라 파일로 받고, 정답이 비면 파일명(A001)에서 읽음 | 모델 미준비 **503** · 모르는 동작 **422** · 파일명에 A001~A006 없음 **422** `NO_LABEL_IN_FILENAME` · 형식 **415** · 100MB 초과 **413** | 효근 (C2) |
| 13 | 분석 시작 | `main.py` | `analyze_file(path, expected)` | 임시 파일 경로 | OpenCV·YOLO가 파일 경로로 영상을 읽기 때문 | 영상 손상 **400** · 사람 없음/너무 짧음 → `"NG"` | 효근 (C2) |
| 14 | 모델 로딩 (서버 시작 시) | `main.py` | `lifespan()` → `load_model()` | joblib + `rf_feature_columns.json` + `rf_training_info.json` | 요청마다 불러오면 느림 + 학습과 같은 특징 이름·순서·개수를 시작 때 검사 | 서버는 켜지고 `/health`에 오류 표시, 예측 요청은 **503** | 효근 (C1) |
| 15 | 관절 추출 | `main.py` | `extract_tracks()` → `frame_features()` | YOLO11 Pose + ByteTrack → 사람별 관절 17개 → 프레임당 38개 | 여러 사람이 있어도 프레임마다 같은 사람 번호 유지 | 사람 없음 → `NO_PERSON` · 가려진 관절은 NaN | 효근 (C1) |
| 16 | 동작 예측 | `main.py` | `predict_track()` → `to_sequence()` → `flatten()` | 30 step × 38 = 1,140 특징 (NaN → 0) → RF `predict_proba` | RF는 입력 길이가 고정이라 30칸으로 맞추고, 5초 구간 확률을 평균해 흔들림 방지 | 쓸 구간이 없으면 `None` → `TOO_SHORT` · 확률 0.55 미만은 '판단 불확실' | 효근 (C1) |
| 17 | OK/NG 판정 | `main.py` | `judge(pred, expected)` | 인식 동작 == 정답 → OK | 규칙이 단순해 결과를 설명하기 쉬움 | 정답이 없으면 verdict `None` (API에서 미리 422로 막음) | 효근 (C1) |
| 18 ★ | 결과 반환 | `main.py` | `analyze_file()` → `_respond()` | JSON `{verdict, status, summary, ...}` | Django가 `verdict`(OK/NG)만 꺼내 저장하도록. 상세(summary)는 나중에 '왜 NG인지' 보여 줄 때 쓸 수 있게 함께 보냄 | `?detail=true`가 없으면 `"OK"`/`"NG"` 글자만 감 | 효근 → 지후 |
| 19 | 결과 받아 적기 | `views.py` | `analyze()` | `res.json()["verdict"]` → `record.verdict`, `status = "DONE"`, `save()` | 다시 계산하면 판정 로직이 Django에도 생겨 중복. 받아 적기만 해야 기준이 바뀌어도 FastAPI만 고침. DB에 남겨야 GET으로 조회 가능 | `FAILED`가 저장되고, JS가 `status !== "DONE"`을 보고 실패 문구 표시 | 지후 |
| 20 | 응답 모양 만들기 | `views.py` | `_result(record)` | `{id, status, verdict}` | POST(분석)·GET(조회) 두 곳이 같은 모양이어야 해서 한 곳에서만 정의. 칸 추가도 여기만 고치면 됨 | 함수 자체는 거의 실패 안 함. 키 이름을 바꾸면 JS가 `verdict`를 못 읽어 실패 문구 | 성환 |
| 21 ★ | 결과 조회 | `screen.js` | `getAnalysisResult(id)` | `fetch` GET `/api/v1/analyze/{id}/` | DB에 실제 저장된 status·verdict를 화면 표시 기준으로 쓰려고 (가이드 STEP 9) | 잘못된 id는 요청 전에 거름. 없는 기록은 Django가 404 → JS가 실패 처리 | 태형 → 성환 |
| 22 | 기록 조회 | `views.py` | `result(request, pk)` ← `analyze/<int:pk>/` | URL 숫자 → `pk` → `get_object_or_404(InferenceRecord, pk=pk)` → `_result()` | DB에 저장된 값을 다시 읽어 화면이 그 결과를 표시. 기록이 없을 때 오류 대신 404를 자동으로 만들어 줌 | 없는 id → **404**, 숫자가 아닌 주소 → 404, GET이 아닌 방식 → **405** | 성환 |
| 23 | 배지 표시 | `screen.js` | `startInference()` | `results.dataset.verdict = verdict` | 서버가 확정한 판정을 화면에 그대로 표시 | FAILED이거나 값이 OK/NG가 아니면 실패 문구. 파일·동작 변경으로 취소한 요청은 배지를 갱신하지 않음 | 태형 → 유나 |
| 24 | 결과 배지 강조 | `screen.html` (CSS) | `.results[data-verdict="OK"] .ok`, `[data-verdict="NG"] .ng` | `data-verdict` 속성 → CSS `outline: 3px` | 두 배지 중 실제 결과를 바로 알아보게 | 속성이 없거나 값이 안 맞으면 둘 다 기본 색으로 남음 | 유나 |

## ★ 팀 약속 (contract)

| 경계 | 보내는 것 | 받는 것 |
|---|---|---|
| 6 · JS → Django | `POST /api/v1/analyze/` — FormData `video`, `action`, 헤더 `X-CSRFToken` | `{id, status, verdict}` |
| 11 · Django → FastAPI | `POST :8001/predict/video?detail=true` — `file`(원래 파일명), `expected_action` | `{verdict: "OK"\|"NG", status, summary, ...}` |
| 21 · JS → Django | `GET /api/v1/analyze/{id}/` | `{id, status, verdict}` (없으면 404) |

## 알아 둘 점 · 개선 후보

- **판단 불가도 NG** — 사람 없음(`NO_PERSON`)·너무 짧음(`TOO_SHORT`)도 FastAPI가 200 + `"NG"`를 줘서 DB에 `DONE / NG`로 저장됨. 화면에서 "자세 틀림"과 "판단 불가"가 구분되지 않음.
- **파일명에 정답이 없는 영상** — 동작을 "파일명에서 확인"으로 두면 422 → `FAILED`. 사용자가 가장 자주 겪을 실패.
- **전처리 검증** — `normalize_keypoints()`는 가정한 방식. 수업 샘플 영상으로 OK·높은 신뢰도가 나오는지 확인 필요.
- **"제목(선택)" 칸** — 화면에만 있고 어디에도 전송·저장되지 않음.
- **다음 확장 (가이드 3장)** — Celery를 붙이면 11·19번(`requests.post` ~ `save()`)이 `views.py`에서 `tasks.py`로 이동.

"""
FastAPI 판정 서버 설정 — 모든 값은 환경변수(.env)로 바꿀 수 있습니다.

왜 설정을 한 파일에 모으나요?
  운동 종류, 판정 기준 각도 같은 값은 팀에서 아직 정하지 않았습니다(미정).
  코드 곳곳에 숫자를 흩어 두면 나중에 하나하나 찾아 고쳐야 하지만,
  여기 모아 두면 이 파일(또는 .env)만 바꾸면 됩니다.
"""
import os

# ── Pose 모델 ─────────────────────────────────────
POSE_WEIGHTS = os.getenv("POSE_WEIGHTS", "yolov8n-pose.pt")  # 없으면 Ultralytics가 자동 다운로드
POSE_CONF = float(os.getenv("POSE_CONF", "0.4"))             # 사람 검출 최소 신뢰도
POSE_TRACKER = os.getenv("POSE_TRACKER", "bytetrack.yaml")    # 사람별 번호(Track ID) 추적기
KEYPOINT_CONF = float(os.getenv("KEYPOINT_CONF", "0.3"))      # 이보다 낮은 관절은 '안 보임'으로 처리

# ── 입력 제한 ─────────────────────────────────────
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "1800"))            # 30fps × 60초
MIN_TRACK_FRAMES = int(os.getenv("MIN_TRACK_FRAMES", "15"))  # 이보다 짧게 잡힌 사람은 제외

# ── 운동 (미정 → 팀 결정 후 수정) ─────────────────
DEFAULT_EXERCISE = os.getenv("DEFAULT_EXERCISE", "squat")  # 요청에 운동 종류가 없을 때 사용
# 영상 전체 판정: 반복 중 OK 비율이 이 값 이상이면 OK
OVERALL_OK_RATIO = float(os.getenv("OVERALL_OK_RATIO", "0.7"))

# ── 스쿼트 판정 기준 (미정 → 예시 값) ─────────────
SQUAT_DOWN_ANGLE = float(os.getenv("SQUAT_DOWN_ANGLE", "120"))  # 무릎 각도가 이보다 작아지면 '앉기 시작'
SQUAT_UP_ANGLE = float(os.getenv("SQUAT_UP_ANGLE", "160"))      # 이보다 커지면 '일어섬' → 1회 완료
SQUAT_DEPTH_MAX = float(os.getenv("SQUAT_DEPTH_MAX", "100"))    # 가장 낮을 때 무릎 각도가 이 이하여야 OK
SQUAT_TORSO_MIN = float(os.getenv("SQUAT_TORSO_MIN", "50"))     # 가장 낮을 때 엉덩이 각도가 이 이상이어야 OK

# ── 푸시업 판정 기준 (미정 → 예시 값) ─────────────
PUSHUP_DOWN_ANGLE = float(os.getenv("PUSHUP_DOWN_ANGLE", "110"))
PUSHUP_UP_ANGLE = float(os.getenv("PUSHUP_UP_ANGLE", "150"))
PUSHUP_DEPTH_MAX = float(os.getenv("PUSHUP_DEPTH_MAX", "90"))     # 가장 낮을 때 팔꿈치 각도
PUSHUP_BODY_MIN = float(os.getenv("PUSHUP_BODY_MIN", "160"))      # 어깨-엉덩이-발목이 일직선(180°)에 가까워야 OK

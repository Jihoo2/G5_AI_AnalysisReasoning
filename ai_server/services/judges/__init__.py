"""
운동 판정기 목록.

새 운동을 추가하는 방법 (예: 런지)
  1) judges/lunge.py 에 ExerciseJudge를 상속한 LungeJudge 작성
  2) 아래 JUDGES 에 한 줄 추가
→ API·응답 형식은 그대로라 다른 팀원 코드는 고칠 필요가 없습니다.
"""
from .base import ExerciseJudge
from .pushup import PushupJudge
from .squat import SquatJudge

JUDGES: dict[str, ExerciseJudge] = {j.key: j for j in (SquatJudge(), PushupJudge())}


def get_judge(key: str) -> ExerciseJudge | None:
    return JUDGES.get(key)

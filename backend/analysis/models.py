from django.db import models


class InferenceRecord(models.Model):
    """AI 분석 요청 1건 = 기록 1줄"""

    STATUS_CHOICES = [
        ("RECEIVED", "접수"),   # 영상 업로드 직후 (STEP 5)
        ("DONE", "완료"),       # FastAPI 결과 저장 후 (STEP 8)
        ("FAILED", "실패"),     # FastAPI 오류 등
    ]
    VERDICT_CHOICES = [
        ("OK", "OK"),
        ("NG", "NG"),
    ]

    video_file = models.FileField(upload_to="videos/")    # 영상은 media/videos/ 에, DB 에는 경로만
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default="RECEIVED")
    verdict = models.CharField(max_length=2, choices=VERDICT_CHOICES, blank=True)  # AI 결과 전엔 빈칸

    def __str__(self):
        return f"#{self.id} {self.status} {self.verdict}"
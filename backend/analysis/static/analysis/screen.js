const byId = (id) => document.getElementById(id);

const video = byId("preview");
const file = byId("file");
const address = byId("address");
const message = byId("message");
const uploadTab = byId("uploadTab");
const cameraTab = byId("cameraTab");

let previewUrl = "";
const results = document.querySelector(".results");
let analysisRequest;

// 선택한 영상 파일을 왼쪽 미리보기에 표시합니다.
function refreshPreview() {
  analysisRequest?.abort();
  delete results.dataset.verdict;
  video.removeAttribute("src");
  video.controls = false;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = "";
  }

  if (!file.hidden && file.files.length > 0) {
    previewUrl = URL.createObjectURL(file.files[0]);
    video.src = previewUrl;
    video.controls = true;
  }

  video.load();
  message.textContent = "";
}

// 선택한 탭에 맞춰 파일 선택 칸 또는 카메라 주소 칸을 표시합니다.
function selectSource(useCamera) {
  file.hidden = useCamera;
  address.hidden = !useCamera;

  uploadTab.setAttribute("aria-pressed", String(!useCamera));
  cameraTab.setAttribute("aria-pressed", String(useCamera));

  refreshPreview();
}

uploadTab.onclick = () => selectSource(false);
cameraTab.onclick = () => selectSource(true);
file.onchange = refreshPreview;

video.onerror = () => {
  message.textContent = "이 브라우저에서 재생할 수 없는 영상입니다.";
};

byId("start").onclick = async () => {
  analysisRequest?.abort();
  delete results.dataset.verdict;

  if (file.hidden) {
    message.textContent = "영상 업로드 탭에서 분석할 영상을 선택해 주세요.";
    return;
  }

  if (!file.files.length) {
    message.textContent = "분석할 영상을 선택해 주세요.";
    file.focus();
    return;
  }

  if (!results.dataset.api) {
    message.textContent = "분석 서버 연결 전입니다.";
    return;
  }

  const request = new AbortController();
  analysisRequest = request;

  const data = new FormData();
  data.append("video", file.files[0]);
  message.textContent = "분석 중입니다.";

  try {
    const response = await fetch(results.dataset.api, {
      method: "POST",
      mode: "same-origin",
      body: data,
      signal: request.signal,
      headers: {
        "X-CSRFToken": document.querySelector("[name=csrfmiddlewaretoken]").value
      }
    });

    if (!response.ok) throw new Error();

    const { confidence } = await response.json();
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      throw new Error();
    }
    if (request.signal.aborted) return;

    results.dataset.verdict = confidence >= 0.8 ? "OK" : "NG";
    message.textContent = "";
  } catch {
    if (!request.signal.aborted) {
      message.textContent = "분석 결과를 받지 못했습니다.";
    }
  }
};
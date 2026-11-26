// 1. 화면 요소와 상태
const uploadTab = document.getElementById("uploadTab");
const cameraTab = document.getElementById("cameraTab");
const videoFile = document.getElementById("videoFile");
const video = document.getElementById("videoPreview");
const camera = document.getElementById("cameraPreview");
const cameraForm = document.getElementById("cameraForm");
const cameraUrl = document.getElementById("cameraUrl");
const cameraStatus = document.getElementById("cameraStatus");

const start = document.getElementById("start");
const action = document.getElementById("action");
const results = document.querySelector(".results");
const analysisStatus = document.getElementById("analysisStatus");
const csrfToken = document.querySelector("[name=csrfmiddlewaretoken]").value;

let fileUrl = "";
let activeRequest = null;


// 2. 기능별 함수

function resetAnalysis() {
  activeRequest?.abort();
  activeRequest = null;

  delete results.dataset.verdict;
  analysisStatus.textContent = "";
  start.disabled = videoFile.hidden || !videoFile.files.length;
}

function switchTab(event) {
  const useCamera = event.currentTarget === cameraTab;
  if (useCamera === videoFile.hidden) return;

  uploadTab.setAttribute("aria-pressed", String(!useCamera));
  cameraTab.setAttribute("aria-pressed", String(useCamera));
  videoFile.hidden = useCamera;
  video.hidden = useCamera;
  cameraForm.hidden = !useCamera;

  video.pause();
  camera.hidden = true;
  camera.removeAttribute("src");
  cameraStatus.textContent = "";
  resetAnalysis();
}

function previewFile() {
  const [file] = videoFile.files;

  if (fileUrl) URL.revokeObjectURL(fileUrl);
  fileUrl = file ? URL.createObjectURL(file) : "";

  if (fileUrl) {
    video.src = fileUrl;
  } else {
    video.removeAttribute("src");
    video.load();
  }

  resetAnalysis();
}

function connectCamera(event) {
  event.preventDefault();
  cameraStatus.textContent = "";
  camera.hidden = false;
  camera.src = cameraUrl.value.trim();
}

function showCameraError() {
  if (!camera.hasAttribute("src")) return;

  camera.hidden = true;
  cameraStatus.textContent = "영상을 열 수 없습니다. 카메라 영상 주소를 확인해 주세요.";
}

async function startInference() {
  const [file] = videoFile.files;
  if (!file || videoFile.hidden || activeRequest) return;

  const request = new AbortController();
  activeRequest = request;
  start.disabled = true;
  delete results.dataset.verdict;
  analysisStatus.textContent = "분석 중입니다.";

  const data = new FormData();
  data.append("video", file);
  data.append("action", action.value);

  try {
    const response = await fetch(results.dataset.api, {
      method: "POST",
      mode: "same-origin",
      headers: { "X-CSRFToken": csrfToken },
      body: data,
      signal: request.signal
    });

    if (!response.ok) throw new Error();

    const { status, verdict } = await response.json();
    if (status !== "DONE" || !["OK", "NG"].includes(verdict)) {
      throw new Error();
    }

    if (request.signal.aborted) return;
    results.dataset.verdict = verdict;
    analysisStatus.textContent = `분석 결과: ${verdict}`;
  } catch {
    if (!request.signal.aborted) {
      analysisStatus.textContent = "분석에 실패했습니다. 서버 상태와 선택한 동작을 확인해 주세요.";
    }
  } finally {
    if (activeRequest === request) {
      activeRequest = null;
      start.disabled = false;
    }
  }
}


// 3. 이벤트 연결
uploadTab.onclick = switchTab;
cameraTab.onclick = switchTab;
videoFile.onchange = previewFile;
cameraForm.onsubmit = connectCamera;
camera.onerror = showCameraError;
action.onchange = resetAnalysis;
start.onclick = startInference;
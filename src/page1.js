const page2_URL = "file:///C:/Users/rhfem/Desktop/class/4-1/%EC%A2%85%ED%95%A9%EC%84%A4%EA%B3%84/code/src/page2.html?";

const inputVideo = document.getElementById('inputVideo'),
    video = document.getElementById('video'),
    submitVideo = document.getElementById('submitVideo');

function submitVideoHandler(event) {
    event.preventDefault();
    if (!video.getAttribute("src")) {
        alert("비디오를 선택해 주세요.");
    } else {
        // 비디오 업로드
        location.href = page2_URL;
        console.log("press upload");
    }
}

function videoInputHandler() {
    const selectedVideo = inputVideo.files[0];
    const videourl = URL.createObjectURL(selectedVideo);
    video.setAttribute("src", videourl);
    console.log(videourl);
}

function init() {
    inputVideo.addEventListener("change", videoInputHandler);
    submitVideo.addEventListener("click", submitVideoHandler);
}

init();
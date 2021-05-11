const page2_URL = "./page2.html?";

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
const page2_URL = "./page2.html?";

const inputVideo = document.getElementById('inputVideo'),
    video = document.querySelector('.video'),
    submitVideo = document.getElementById('submitVideo');

function uploadVideo() {
    window.localStorage.setItem("videoUrl", video.getAttribute("src"));
}

function submitVideoHandler(event) {
    event.preventDefault();
    if (!video.getAttribute("src")) {
        alert("비디오를 선택해 주세요.");
    } else {
        // 비디오 업로드
        uploadVideo();
        location.href = page2_URL;
        console.log("press upload");
    }
}

function videoInputHandler() {
    const selectedVideo = inputVideo.files[0];
    const videoUrl = URL.createObjectURL(selectedVideo);
    console.log(video);
    video.setAttribute("src", videoUrl);
}

function init() {
    inputVideo.addEventListener("change", videoInputHandler);
    submitVideo.addEventListener("click", submitVideoHandler);
}

init();
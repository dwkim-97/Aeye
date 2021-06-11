const loading_URL = "./loading.html?";

const inputVideoBtn = document.getElementById('inputVideoBtn'),
    video = document.querySelector('.video'),
    submitVideoBtn = document.getElementById('submitVideoBtn');

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
        location.href = loading_URL;
        console.log("press upload");
    }
}

function videoInputHandler() {
    const selectedVideo = inputVideoBtn.files[0];
    const videoUrl = URL.createObjectURL(selectedVideo);
    video.style.height = "80%";
    console.log(video);
    video.setAttribute("src", videoUrl);
}

function videoClickHandler() {
    console.log(video.dataset.isPlay);
    if (video.dataset.isPlay === "playing") {
        video.pause();
        video.dataset.isPlay = "paused";
    } else {
        video.play();
        video.dataset.isPlay = "playing";
    }
}

function init() {
    inputVideoBtn.addEventListener("change", videoInputHandler);
    video.addEventListener("click", videoClickHandler);
    submitVideoBtn.addEventListener("click", submitVideoHandler);
}


init();
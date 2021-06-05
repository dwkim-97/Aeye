const page4_URL = "./page4.html";

const inputVideoBtn = document.getElementById('inputVideoBtn'),
    video = document.querySelector('.video'),
    submitVideoBtn = document.getElementById('submitVideoBtn'),
    goBackBtn = document.getElementById("goBackBtn"),
    selectedVideos = document.getElementById('selectedVideos');


function goBackHandler() {
    history.back();
}

function uploadVideo() {
    window.localStorage.setItem("videoUrl", video.getAttribute("src"));
}

function submitVideoHandler(event) {
    event.preventDefault();
    if (!selectedVideos.childElementCount) {
        alert("비디오를 선택해 주세요.");
    } else {
        // 비디오 업로드
        // uploadVideo();
        //location.href = page4_URL;
        console.log("press upload");
    }
}

// 비디오는 우선 최대 9개, 변경 필요
function videoInputHandler() {
    for (let i = 0; i < 9; i++) {
        const selectedVideo = inputVideoBtn.files[i];
        const videoUrl = URL.createObjectURL(selectedVideo);

        const newLi = document.createElement('li');
        newLi.setAttribute("id", "videoBox");
        selectedVideos.appendChild(newLi);


        const newVideo = document.createElement('video');
        newVideo.setAttribute("src", videoUrl);
        newVideo.setAttribute("data-is-play", "paused");
        newVideo.addEventListener("click", videoClickHandler);
        newLi.appendChild(newVideo);
    }
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

// 페이지2에서 체크한 이미지
function getCheckedImg() {
    const checkedImg = document.getElementById("checkedImg");
    checkedImg.setAttribute("src", window.localStorage.getItem("checkSrc"));
}

function init() {
    getCheckedImg();
    goBackBtn.addEventListener('click', goBackHandler);
    inputVideoBtn.addEventListener("change", videoInputHandler);
    submitVideoBtn.addEventListener("click", submitVideoHandler);
}


init();
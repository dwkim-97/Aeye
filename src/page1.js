const loading_URL = "./loading.html?";

const inputVideoBtn = document.getElementById('inputVideoBtn'),
    videoBlock = document.querySelector('.videoBlock'),
    submitVideoBtn = document.getElementById('submitVideoBtn'),
    inputImageBtn = document.getElementById('inputImageBtn'),
    imageBlock = document.querySelector('.imageBlock'),
    submitImageBtn = document.getElementById('submitImageBtn');

let video, image;

function uploadVideo() {
    window.localStorage.setItem("videoUrl", video.getAttribute("src"));
}

function uploadImage() {
    window.localStorage.setItem("imageUrl", image.getAttribute("src"));
}

function submitVideoHandler(event) {
    event.preventDefault();
    if (!video) {
        alert("비디오를 선택해 주세요.");
    } else {
        // 비디오 업로드
        uploadVideo();
        if (image) {
            location.href = loading_URL;
        }
    }
}

function videoInputHandler() {
    const selectedVideo = inputVideoBtn.files[0];
    const videoUrl = URL.createObjectURL(selectedVideo);
    video = document.createElement("video");
    video.setAttribute("data-is-play", "paused");
    video.className = "video";
    video.setAttribute("src", videoUrl);
    video.addEventListener("click", videoClickHandler);
    videoBlock.appendChild(video);
}

function videoClickHandler() {
    if (video.dataset.isPlay === "playing") {
        video.pause();
        video.dataset.isPlay = "paused";
    } else {
        video.play();
        video.dataset.isPlay = "playing";
    }
}

function imageInputHandler() {
    const selectedImage = inputImageBtn.files[0];
    const imageUrl = URL.createObjectURL(selectedImage);
    image = document.createElement("img");
    image.className = "image";
    image.setAttribute("src", imageUrl);
    imageBlock.appendChild(image);
}

function submitImageHandler(event) {
    event.preventDefault();
    if (!image) {
        alert("이미지를 선택해 주세요.");
    } else {
        // 이미지 업로드
        uploadImage();
        if (video) {
            location.href = loading_URL;
        }
    }
}

function init() {
    inputVideoBtn.addEventListener("change", videoInputHandler);
    submitVideoBtn.addEventListener("click", submitVideoHandler);
    inputImageBtn.addEventListener("change", imageInputHandler);
    submitImageBtn.addEventListener("click", submitImageHandler);
}


init();
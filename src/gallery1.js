const gallery2_URL = "./gallery2.html?";
const test_data_URL = "../test_data";

const goBackBtn = document.getElementById("goBackBtn"),
    goNextBtn = document.getElementById('goNextBtn'),
    videoBlock = document.getElementById('videoBlock'),
    video = document.querySelector('#videoPlayer'),
    closestImageBlock = document.querySelector('#closestImageBlock'),
    galleryBlock = document.getElementById('galleryBlock'),
    peopleGallery = document.getElementById('peopleGallery');

let player;

// 이전으로
function goBackHandler() {
    history.back();
}

// 다음으로
function goNextHandler() {
    location.href = gallery2_URL;
}

// 특정 시간으로 비디오 재생하기
function playVideoWithTime(event) {
    const time = event.target.dataset.time;
    video.currentTime = time;
    video.play();
    video.dataset.isPlay = "playing";
}

// 비디오 페이지 넘어오게 하기 - 서버 연동시 변경
function getOriginalVideo() {
    //video.setAttribute('src', "../test_data/video.mp4");
    // console.log(video);
    //video.setAttribute('poster', "../test_data/video_poster.png");

    player = videojs("videoPlayer", {
        sources: [
            { src: "../test_data/video.mp4", type: "video/mp4" }
        ],
        controls: true,
        playsinline: true,
        muted: true,
        preload: "metadata",
    });
}

// 인물 갤러리 생성 - 서버 연동시 변경
function createGallery() {
    let index = 0;
    while (index < 5) {
        const src = test_data_URL + '/represent_persons/group_' + index + '.jpg';
        const li = document.createElement('li');
        li.id = index;
        const personBlock = createPersonBlock(src);
        li.appendChild(personBlock);
        peopleGallery.appendChild(li);
        index++;
    }
}

function changeSecondsToTime(seconds) {
    const time = {};
    time.seconds = seconds % 60 > 9 ? seconds % 60 : `0${seconds % 60}`;
    seconds = Math.floor(seconds / 60);

    time.minutes = seconds % 60 > 9 ? seconds % 60 : `0${seconds % 60}`;
    seconds = Math.floor(seconds / 60);

    time.hours = seconds > 9 ? seconds : `0${seconds}`;

    return `${time.hours}:${time.minutes}:${time.seconds}`;
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

function createPersonBlock(src, isBold = false) {
    const fileName = src.split('/').pop();

    const imageBlockElement = document.createElement('div');
    imageBlockElement.className = "image_block";

    const imageElement = document.createElement("img");
    imageElement.setAttribute('src', src);
    imageElement.setAttribute("class", "person_img");

    const imageName = document.createElement(isBold ? 'strong' : 'p');
    imageName.className = "image_name";
    imageName.innerText = fileName;

    imageBlockElement.append(imageElement, imageName);

    return imageBlockElement;
}

function setClosestImage() {
    const closestImageElement = createPersonBlock(test_data_URL + '/matching_result/matching_person.jpg', true);
    closestImageElement.id = "closest_image";
    closestImageBlock.appendChild(closestImageElement);
}

function init() {
    getOriginalVideo();
    createGallery();
    goBackBtn.addEventListener('click', goBackHandler);
    goNextBtn.addEventListener('click', goNextHandler);
    video.addEventListener('click', videoClickHandler);
    setClosestImage();
}

init();
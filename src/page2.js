const page3_URL = "./page3.html?";

const goBackBtn = document.getElementById("goBackBtn"),
    goNextBtn = document.getElementById('goNextBtn'),
    videoBlock = document.getElementById('videoBlock'),
    video = document.querySelector('.video'),
    galleryBlock = document.getElementById('galleryBlock'),
    peopleGallery = document.getElementById('peopleGallery');



// 이전으로
function goBackHandler() {
    history.back();
}

// 다음으로
function goNextHandler() {
    location.href = page3_URL;
}

// 특정 시간으로 비디오 재생하기
function playVideoWithTime(event) {
    const time = event.target.innerText;
    video.currentTime = time;
    video.play();
    video.dataset.isPlay = "playing";
    console.log(video.dataset.isPlay);
}

// 비디오 페이지 넘어오게 하기 - 서버 연동시 변경
function getOriginalVideo() {
    video.setAttribute('src', "../test_data/52.mp4");
    video.setAttribute('poster', "../test_data/video_poster.png");
}

// 인물 갤러리 생성 - 서버 연동시 변경
function getGallery() {
    for (let i = 0; i < 7; i++) {
        const TEMP_LIST_ITEM = document.createElement('li');
        const TEMP_PERSON_OBJECT = document.createElement('img');
        const TEMP_TIME = document.createElement('p');

        galleryBlock.appendChild(TEMP_LIST_ITEM);
        TEMP_LIST_ITEM.appendChild(TEMP_PERSON_OBJECT);
        TEMP_LIST_ITEM.appendChild(TEMP_TIME);

        TEMP_PERSON_OBJECT.setAttribute('src', `../test_data/${i}.png`);

        TEMP_TIME.innerText = `0${i}`;
        TEMP_TIME.addEventListener('click', playVideoWithTime)
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

function init() {
    getOriginalVideo();
    getGallery();
    goBackBtn.addEventListener('click', goBackHandler);
    goNextBtn.addEventListener('click', goNextHandler);
    video.addEventListener('click', videoClickHandler);
}

init();
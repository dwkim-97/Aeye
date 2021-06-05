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


// 사람 이미지 선택해서 체크
function personObjectCheckHandler(event) {
    console.log(event.target);
    const clickedLi = event.target.parentNode;
    clickedLi.setAttribute('checked', "true");
    clickedLi.style.border = "solid";
    clickedLi.style.borderColor = "red";
    // 이전에 있던 체크드 지우기
    if (window.localStorage.getItem("checkId")) {
        const oldCheckId = window.localStorage.getItem("checkId");
        const oldCheckedLi = document.getElementById(oldCheckId);
        oldCheckedLi.setAttribute('checked', "false");
        oldCheckedLi.style.border = "none";
    }
    window.localStorage.setItem("checkId", clickedLi.id);
    window.localStorage.setItem("checkSrc", event.target.src);
}

// 인물 갤러리 생성 - 서버 연동시 변경
function getGallery() {
    for (let i = 0; i < 7; i++) {
        const TEMP_LIST_ITEM = document.createElement('li');
        const TEMP_PERSON_OBJECT = document.createElement('img');
        const TEMP_TIME = document.createElement('p');

        peopleGallery.appendChild(TEMP_LIST_ITEM);
        TEMP_LIST_ITEM.appendChild(TEMP_PERSON_OBJECT);
        TEMP_LIST_ITEM.appendChild(TEMP_TIME);
        TEMP_LIST_ITEM.setAttribute('id', i);
        TEMP_LIST_ITEM.setAttribute('checked', "false");

        TEMP_PERSON_OBJECT.setAttribute('src', `../test_data/${i}.png`);
        TEMP_PERSON_OBJECT.addEventListener('click', personObjectCheckHandler);

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
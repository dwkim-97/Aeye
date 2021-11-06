const page3_URL = "./page3.html?";

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
    location.href = page3_URL;
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


// // 사람 이미지 선택해서 체크
// function personObjectCheckHandler(event) {
//     const clickedLi = event.target;
//     console.log(clickedLi);
//     clickedLi.setAttribute('checked', "true");
//     clickedLi.style.border = "solid";
//     clickedLi.style.borderColor = "red";
//     // 이전에 있던 체크드 지우기
//     if (window.localStorage.getItem("checkId")) {
//         const oldCheckId = window.localStorage.getItem("checkId");
//         const oldCheckedLi = document.getElementById(oldCheckId);
//         oldCheckedLi.setAttribute('checked', "false");
//         oldCheckedLi.style.border = "none";
//     }
//     window.localStorage.setItem("checkId", clickedLi.id);
//     window.localStorage.setItem("checkSrc", event.target.id);
// }

// 인물 갤러리 생성 - 서버 연동시 변경
// i값을 represent_person안에 있는 파일수 즉 클러스터만큼 하면 됨
function getGallery() {
    for (let i = 0; i < 5; i++) {
        const TEMP_LIST_ITEM = document.createElement('li');
        const TEMP_LIST_ITEM_CONTAINOR = document.createElement("div");
        const TEMP_PERSON_OBJECT = document.createElement('img');
        const TEMP_TIME_BLOCK = document.createElement('div');

        peopleGallery.appendChild(TEMP_LIST_ITEM);
        TEMP_LIST_ITEM.appendChild(TEMP_LIST_ITEM_CONTAINOR);
        TEMP_LIST_ITEM.setAttribute('id', i);
        TEMP_LIST_ITEM.setAttribute("class", "list_item");
        //TEMP_LIST_ITEM.addEventListener('click', personObjectCheckHandler);

        TEMP_LIST_ITEM_CONTAINOR.appendChild(TEMP_PERSON_OBJECT);
        TEMP_LIST_ITEM_CONTAINOR.setAttribute("class", "person");
        TEMP_LIST_ITEM_CONTAINOR.setAttribute('checked', "false");

        TEMP_LIST_ITEM.appendChild(TEMP_TIME_BLOCK);

        TEMP_PERSON_OBJECT.setAttribute('src', `../test_data/represent_persons/group_${i}.jpg`);
        TEMP_PERSON_OBJECT.setAttribute("class", "person_img");


        TEMP_TIME_BLOCK.setAttribute("class", "time_box");

        if (i === 0) {
            [0, 63, 74, 269, 313].forEach((aTime) => {
                const TEMP_TIME = document.createElement('p');
                TEMP_TIME.innerText = changeSecondsToTime(aTime);
                TEMP_TIME.dataset.time = aTime;
                TEMP_TIME.addEventListener('click', playVideoWithTime)
                TEMP_TIME_BLOCK.appendChild(TEMP_TIME);
            })
        } else {
            for (let j = 0; j < 5; j++) {
                const TEMP_TIME = document.createElement('p');
                const RANDOM_TIME = Math.floor(Math.random() * 403);
                TEMP_TIME.innerText = changeSecondsToTime(RANDOM_TIME);
                TEMP_TIME.dataset.time = RANDOM_TIME;
                TEMP_TIME.addEventListener('click', playVideoWithTime)
                TEMP_TIME_BLOCK.appendChild(TEMP_TIME);
            }
        }
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

function setClosestImage() {
    const closestImageElement = peopleGallery.children[5].cloneNode(true);
    closestImageBlock.appendChild(closestImageElement);
}

function init() {
    getOriginalVideo();
    getGallery();
    goBackBtn.addEventListener('click', goBackHandler);
    goNextBtn.addEventListener('click', goNextHandler);
    video.addEventListener('click', videoClickHandler);
    setClosestImage();
}

init();

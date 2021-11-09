const META_FILE_NAME = "meta.txt";

const goBackBtn = document.getElementById("goBackBtn"),
    frameGallery = document.getElementById("frameGallery");

function goBackHandler() {
    history.back();
}

function composeFrameGallery() {
    for (let i = 0; i < 5; i++) {
        const frameName = `result_frame_${i}.jpg`
        const frameBlockElement = document.createElement('div');
        frameBlockElement.id = i;
        frameBlockElement.className = "frame_block";
        const frame = document.createElement('img');
        frame.setAttribute('src', "../test_data/target_appear_frame/" + frameName);
        const frameNameElement = document.createElement('p');
        frameNameElement.innerText = frameName;
        frameBlockElement.append(frame, frameNameElement);
        frameGallery.appendChild(frameBlockElement);
    }
}

function init() {
    goBackBtn.addEventListener('click', goBackHandler);
    composeFrameGallery();
}


init();
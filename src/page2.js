const goBackBtn = document.getElementById("goBackBtn");

function goBackHandler() {
    history.back();
}

function init() {
    goBackBtn.addEventListener('click', goBackHandler)
}

init();
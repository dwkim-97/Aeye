const page1_URL = "file:///C:/Users/rhfem/Desktop/class/4-1/%EC%A2%85%ED%95%A9%EC%84%A4%EA%B3%84/code/src/page1.html?";

function checkUid(uid) {
    const UID_FROM_SERVER = '0000';
    if (uid == UID_FROM_SERVER) {
        return true;
    } else {
        alert("올바르지 않은 ID 입니다.");
        return false;
    }
}

function onSubmit(event) {
    event.preventDefault();
    const uid = document.getElementById('uid');
    if (checkUid(uid.value)) {
        location.href = page1_URL;
    } else {
        uid.value = "";
    }
}

function init() {
    document.getElementById('submit').addEventListener('click', onSubmit);
}

init()
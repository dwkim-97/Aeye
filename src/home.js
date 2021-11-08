const gallery1_URL = "./gallery1.html?";

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
        location.href = gallery1_URL;
    } else {
        uid.value = "";
    }
}

function init() {
    document.getElementById('submit').addEventListener('click', onSubmit);
}

init()
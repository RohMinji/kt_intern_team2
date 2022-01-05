const video = document.getElementById('course-player');

/* play button event test */
function videoPlay() { 
    video.play(); 
};

/* pause button event test */
function videoPause() { 
    video.pause(); 
};

function nextPagee() {
    location.href = NEXT_URL;
}

// Index Page SY detection
$('#start-btn').click(function (e) {
    e.preventDefault();
    $.ajax({
        url: SY_DETECT_URL,
        dataType: "json",
        success: function (data) {
            if (data.SY_COUNT >= 100) {
                //alert("안녕하세요 승용님, 학습을 시작하겠습니다.");
                location.href = NEXT_URL;
            }
            else {
                console.log(data.SY_COUNT);
                alert("학습자가 인식되지 않았습니다. 다시 시도해주세요.");
            }
        },
        error: function (error) {
            alert("에러가 발생했습니다.");
        }
    });
});

// No Student Detect => Course STOP
function videoControl(){
    $.ajax({
        url: SY_EXIST_URL,
        method: "GET",
        dataType: "json",
        success: function (data) {
            console.log(data)
            if (data.videoValue == 1 || data.sy_exist == 0) {
                videoPause()
            } 
            else {
                videoPlay()
            }
        },
        error: function (error) {
            alert("에러가 발생했습니다.");
        }
    });
};

$(document).ready(function(){
    setInterval(videoControl, 1000);
})

function nextpagecontrol(){
    $.ajax({
        url: SY_DETECT_URL,
        method: "GET",
        dataType: "json",
        success: function (data) {
            console.log(data)
            if (data.SY_COUNT >= 100 && data.SY_COUNT<=115) {
                alert("안녕하세요 승용님, 학습을 시작하겠습니다.")
                nextPagee()
            }
        },
        error: function (error) {
            alert("에러가 발생했습니다.");
        }
    });
};

$(document).ready(function(){
    setInterval(nextpagecontrol, 1000);
})
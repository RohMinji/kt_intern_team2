from django.shortcuts import render
from models.sleep_detection import eyesize_dic, empty_dic

# Create your views here.
def course(request):
    return render(request, 'courses/course.html')

def result(request):
    from models.sleep_detection import YAWN_COUNTER
    from models.dance_detection import avg_score
    from core.views import client_socket

    client_socket.sendall(("수고하셨습니다 승용님, 학습을 종료하겠습니다." + "오늘 수업을 듣는 동안 총 " + 
        str(YAWN_COUNTER) + "번 하품 하셨고. " + str(list(empty_dic.values()).count(1))
        + " 초 자리 비우셨습니다.").encode())

    try:
        avg_score = int(avg_score)
    except:
        avg_score = "No Score"
        
    context = {
        "key": list(eyesize_dic.keys()),
        "val": list(eyesize_dic.values()),
        "cnt_empty":list(empty_dic.values()).count(1),
        "YAWN_COUNTER": YAWN_COUNTER,
        "avg_score": avg_score,
    }
    return render(request, "courses/result.html", context)
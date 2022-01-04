from django.shortcuts import render
from models.sleep_detection import eyesize_dic, empty_dic

# Create your views here.
def course(request):
    return render(request, 'recognitions/course.html')

def result(request):
    from models.sleep_detection import YAWN_COUNTER
    from models.dance_detection import avg_score
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
    return render(request, "recognitions/result.html", context)
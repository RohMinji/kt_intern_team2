from django.shortcuts import render
from models.sleep_detection import img_list,txt_list

######왜 안 되는지 모르겠음!
#from dance_detection.DANCE_DETECTION_sample2 import avg_score

# Create your views here.
def course(request):
    return render(request, 'recognitions/course.html')

def result(request):
    # stripped = [w.strip() for w in list(img_list.keys())]
    print("key", list(img_list.keys()))
    print("val", list(img_list.values()))
    context = {
        "key": list(img_list.keys()),
        "val": list(img_list.values()),
        "cnt_yawn":list(txt_list.values()).count(0),
        "cnt_empty":list(txt_list.values()).count(1)
        #,"avg_score": avg_score
    }
    return render(request, "recognitions/result.html", context)
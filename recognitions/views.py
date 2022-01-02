from django.shortcuts import render

# Create your views here.
def course(request):
    return render(request, 'recognitions/course.html')

def result(request):
    return render(request, "recognitions/result.html")
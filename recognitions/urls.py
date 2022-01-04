from django.urls import path, include
from recognitions import views

app_name = "recognitions"

urlpatterns = [
    path('course/', views.course, name="course"),
    path('result/', views.result, name="result"),
]
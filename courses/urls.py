from django.urls import path
from courses import views

app_name = "courses"

urlpatterns = [
    path('course/', views.course, name="course"),
    path('result/', views.result, name="result"),
]
from django.urls import path
from courses import views

app_name = "courses"

urlpatterns = [
    path('', views.course, name="course"),
    path('result/', views.result, name="result"),
]
from django.urls import path, include
from recognitions import views
from rest_framework import routers
from rest_framework_swagger.views import get_swagger_view

import recognitions.api

app_name = "recognitions"

router = routers.DefaultRouter()
router.register('recognitions', recognitions.api.ModelNameViewSet, basename="recognitions")

urlpatterns = [
    path('course/', views.course, name="course"),
    path('api/doc', get_swagger_view(title='Rest API Document')),
    path('api/v1/', include((router.urls, 'recognitions'), namespace="api")),
]
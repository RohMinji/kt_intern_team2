from recognitions.models import ModelName
from rest_framework import serializers, viewsets

class ModelNameSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelName
        fields = '__all__'

class ModelNameViewSet(viewsets.ModelViewSet):
    quaryset = ModelName.objects.all()
    serializer_class = ModelNameSerializer
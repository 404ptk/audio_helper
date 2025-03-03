from django.urls import path
from .views import musichelper_view

urlpatterns = [
    path('', musichelper_view, name='musichelper'),
]

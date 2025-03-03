from django.urls import path
from .views import musichelper_view, analyze_track_view

urlpatterns = [
    path('', musichelper_view, name='musichelper'),
    path('analyze/', analyze_track_view, name='analyze_track'),
]

# youtube_blog/urls.py

from django.urls import path
from youtube_blog import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home, name='home'),  # Define a path for the root URL (index)
    path('generate-blog-and-quiz/', views.generate_blog_and_quiz, name='generate_blog_and_quiz'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
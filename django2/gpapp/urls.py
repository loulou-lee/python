#django2에 있는 urls.py에서 위힘한 url이다

from gpapp import views
from django.urls.conf import path
urlpatterns = [
    path('insert', views.insertFunc), # Function views,
    path('insertprocess', views.insertprocessFunc),
]
"""django9_board URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path
from myboard.views import view1, view2


urlpatterns = [
   
    path('list', view1.listFunc),
    
    path('insert', view1.insertFunc),  
    path('insertok', view1.insertokFunc),
    
    path('search', view1.searchFunc),
   
    path('content', view1.contentFunc),
    
    path('update', view1.updateFunc),  
    path('updateok', view1.updateokFunc),
    
    path('delete', view1.deleteFunc),  
    path('deleteok', view1.deleteokFunc),
    
    # 답글
    path('reply', view2.replyFunc),  
    path('replyok', view2.replyokFunc),
]

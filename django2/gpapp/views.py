from django.shortcuts import render
from django.views.generic.base import TemplateView

# Create your views here.
def mainFunc(request):
    return render(request, 'index.html')

class CallView(TemplateView):
    template_name = "callget.html"
    
def insertFunc(request):
    return render(request, 'insert.html')

def insertprocessFunc(request):
    if request.method == 'Post':
        name = request.GET.get("name") # java : request.getParameter("name")
        #java에서는 get과 post방식 똑같이 받지만 django는 다르게 받는다
        print(name)
        return render(request, 'list.html', {'myname':name, })

def insertFunc2(request):
    if request.method == 'GET':
        return render(request, 'insert2.html')
    elif request.method == 'POST':
        name = request.Post.get("name")
        return render(request, 'list.html', {'myname':name})
    else:
        print('요청 에러')
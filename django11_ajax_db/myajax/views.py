from django.shortcuts import render
from myajax.models import Sangdata
from django.http.response import HttpResponse
import json

# Create your views here.
def MainFunc(request):
    return render(request, 'main.html')

def ListFunc(request):
    return render(request, 'list.html')

def ListDbFunc(request):
    sdata = Sangdata.objects.all()
    
    datas = []
    for s in sdata:
        dic = {'code':s.code, 'sang':s.sang, 'su':s.su,'dan':s.dan}
        datas.append(dic)
    print(datas)
    return HttpResponse(json.dumps(datas), content_type="application/json")

def goodFunc(request):
    kor = 100
    end =100
    return render(request, 'show.html')
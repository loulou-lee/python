from django.shortcuts import render
import time
import json
from django.http.response import HttpResponse
# Create your views here.

lan = {
    'id':123,
    'name':'파이썬',
    'history':[
        {'date':'2022-9-20', 'exam':'basic'},
        {'date':'2022-10-20', 'exam':'django'}, 
    ]    
}

def testFunc():
    print(type(lan)) # <class 'dict'>
    
    # JSON 인코딩 : Python Object(dict, list, tuple...)를 JSON 문자열로 변경
    # jsonString = json.dumps(lan)
    jsonString = json.dumps(lan, indent=4)
    print(jsonString)
    print(type(jsonString)) # <class 'str'>
    
    # JSON 디코딩 : JSON 문자열을 Python Object(dict, list, tuple...)로 변경
    dic = json.loads(jsonString)
    print(type(dic)) # <class 'dict'>
    print(dic)
    print(dic['name'])

def indexFunc(request):
    testFunc()
    return render(request, 'abc.html')

def Func1(request):
    msg = request.GET.get('msg')
    msg = "nice" + msg
    print(msg)
    context = {'key':msg}
    time.sleep(10);
    return HttpResponse(json.dumps(context), content_type="application/json")

def Func2(request):
    datas = [
        {'irum':'홍길동', 'nai':22},
        {'irum':'고길동', 'nai':32},
        {'irum':'신길동', 'nai':42},    
    ]
    
    return HttpResponse(json.dumps(datas), content_type="application/json")
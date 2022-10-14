from django.shortcuts import render
from django.http.response import HttpResponse

# Create your views here.
def indexFunc(request):
    '''
    msg = "장고 만세?"
    ss = "<html><body>장고 프로젝트 처리 %s</body></html>"%msg
    # return HttpResponse('요청 처리')
    return HttpResponse('ss')
    '''
    
    # 클라이언트에게 html 파일을 반환 - 파이썬 값을 html에 담아서 전달
    msg = "hi django"
    context = {'msg':msg} # dict type으로 작성해 html 문서에 기술한 장고 template 기호와 매핑
    
    return render(request, 'main.html', context) # forward 방식 기본

def helloFunc(request):
    return render(request, 'show.html')
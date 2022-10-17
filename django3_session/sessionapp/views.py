from django.shortcuts import render
from django.http.response import HttpResponseRedirect

# Create your views here.
def mainFunc(request):
    return render(request,'main.html')

def setOsFunc(request):
    #요청방식이 GET이고 파라미터 name이 "favorite_os"일때
    if "favorite_os" in request.GET:
        #print(request.GET.favorite_os)
        print(request.GET["favorite_os"]) #이것을 더 많이 사용
        
        #"f_os"라는 키로 session 생성
        request.session["f_os"]=request.GET["favorite_os"]
        #return render()형식은 forwarding이기 때문에 클라이언트를 통한 요청이 불가능하다 (요청경로가 바뀌지 않기때문)
        #다시 말해 메인 urls.py를 만날 수 없다
        
        #forwarding 말고 redirect 방식을 사용한다면 메인 urls.py을 만날 수 있다 그래서 클라이언트를 통한 요청이 가능하다
        return HttpResponseRedirect("/showos")
    else:
        return render(request,'selectos.html') # 요청값에 "favorite_os"이 없는 경우

def showOsFunc(request):
    # print('여기까지 도착')
    dict_context = {}
    
    if "f_os" in request.session: # 세션 값 중에 "f_os"가 있으면 처리
        print('유효 시간 : ', request.session.get_expiry_age())
        dict_context['sel_os'] = request.session["f_os"]
        dict_context['message'] = "그대가 선택한 운영체제는 %s"%request.session["f_os"]
    else:
        dict_context['message'] = "운영체제를 선택하지 않았군요"
        
    # del request.session["f_os"] # 특정 세션 삭제
    request.session.set_expiry(5) # 5초 후 세션 삭제
    # set_expiry(0) 브라우저가 닫힐 때 세션이 해제됨
    
    return render(request, 'show.html', dict_context)
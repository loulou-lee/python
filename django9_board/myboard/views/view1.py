from django.shortcuts import render, redirect
from myboard.models import BoardTab
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from datetime import datetime

# Create your views here.
def mainFunc(request):
    aa = "<div><h2>게시판 메인</h2></div>"   
    return render(request, 'boardmain.html', {'msg': aa})

def listFunc(request):
    # data_all = BoardTab.objects.all().order_by('-id') # 댓글 코드 X
    data_all = BoardTab.objects.all().order_by('-gnum', 'onum')
    
    paginator = Paginator(data_all, 5)
    page = request.GET.get('page')
    try:
        datas = paginator.page(page)
    except PageNotAnInteger:
        datas = paginator.page(1)
    except EmptyPage:
        datas = paginator.page(paginator.num_pages)
        
    return render(request, 'board.html', {'datas':datas})
    
def insertFunc(request):
    return render(request, 'insert.html')
    
def insertokFunc(request):
    if request.method == 'POST':
        try:
            BoardTab()
            gbun = 1 # Group number 구하기
            datas = BoardTab.objects.all()
            if datas.count() != 0:
                gbun = BoardTab.objects.latest('id').id + 1
            BoardTab(
                name = request.POST.get('name'),
                passwd = request.POST.get('passwd'),
                mail = request.POST.get('mail'),
                title = request.POST.get('title'),
                cont = request.POST.get('cont'),
                bip = request.META['REMOTE_ADDR'],
                bdate = datetime.now(),
                readcnt = 0,
                gnum = gbun,
                onum = 0,
                nested = 0
            ).save()
        except Exception as e:
            print('insert err : ', e)
            return render(request, 'error.html')
        
    return redirect('/board/list') # 추가 후 목록 보기

def searchFunc(request):
    if request.method == 'POST':
        s_type = request.POST.get('s_type')
        s_value = request.POST.get('s_value')
        # print(s_type, s_value)
        # SQL의 like 연산 --> ORM에서는 __contains=값
        if s_type == 'title':
            datas_search=BoardTab.objects.filter(title__contains=s_value).order_by('-id')
        elif s_type == 'name':
            datas_search=BoardTab.objects.filter(name__contains=s_value).order_by('-id')
            
        paginator = Paginator(datas_search, 5)
        page = request.GET.get('page')
        
        try:
            datas = paginator.page(page)
        except PageNotAnInteger:
            datas = paginator.page(1)
        except EmptyPage:
            datas = paginator.page(paginator.num_pages)
            
        return render(request, 'board.html', {'datas':datas})
    
def contentFunc(request):
    page = request.GET.get('page')
    data = BoardTab.objects.get(id=request.GET.get('id'))
    data.readcnt = data.readcnt + 1 # 조회수 증가
    data.save() # 조회수 update
    return render(request, 'content.html', {'data_one':data, 'page':page})

def updateFunc(request):
    try:
        data = BoardTab.objects.get(id=request.GET.get('id'))
    except Exception as e:
        return render(request, 'error.html')
        
    return render(request, 'update.html', {'data_one':data})
    
def updateokFunc(request): # 수정 처리
    try:
        upRec = BoardTab.objects.get(id=request.POST.get('id'))
        
        # 비밀번호 비교 후 수정 여부 결정
        if upRec.passwd == request.POST.get('up_passwd'):
            upRec.name = request.POST.get('name')
            upRec.mail = request.POST.get('mail')
            upRec.title = request.POST.get('title')
            upRec.cont = request.POST.get('cont')
            upRec.save()
        else:
            return render(request, 'update.html', {'data_one':upRec, 'msg':'비밀번호 불일치'})
            
        
    except Exception as e:
        return render(request, 'error.html')
     
    return redirect('/board/list') # 수정 후 목록 보기

def deleteFunc(request):
    try:
        del_data = BoardTab.objects.get(id=request.GET.get('id'))
    except Exception as e:
        return render(request, 'error.html')
        
    return render(request, 'delete.html', {'data_one':del_data})
    
def deleteokFunc(request):
    del_data = BoardTab.objects.get(id=request.POST.get('id'))
    
    if del_data.passwd == request.POST.get('del_passwd'):
        del_data.delete();
        return redirect('/board/list') # 삭제 후 목록 보기
    else:
        return render(request, 'error.html')
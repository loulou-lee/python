from django.shortcuts import render, redirect
from myguest.models import Guest
from django.http.response import HttpResponseRedirect
from datetime import datetime
from django.utils import timezone

# Create your views here.
def MainFunc(request):
    msg = "<h1>홈페이지</h1>"
    return render(request, 'main.html', {'msg':msg})

def ListFunc(request):
    
    gdatas = Guest.objects.all()
    '''
    print(gdatas)
    print(Guest.objects.get(id=1))
    print(Guest.objects.filter(id=1))
    print(Guest.objects.filter(title='안녕'))
    print(Guest.objects.filter(title__contains='안녕'))
    #...
    gdatas = Guest.objects.all().order_by('title') #오름차순
    gdatas = Guest.objects.all().order_by('-title') #내림차순
    
    gdatas = Guest.objects.all().order_by('-id')
    gdatas = Guest.objects.all().order_by('title', '-id')
    gdatas = Guest.objects.all().order_by('-id')[0:2]
    '''
    return render(request, 'list.html', {'gdatas':gdatas})

def InsertFunc(request):
    return render(request, 'insert.html')

def InsertOkFunc(request):
    if request.method == 'POST':
        # print(request.POST.get('title'))
        # print(request.POST['title'])
        Guest(
            title = request.POST['title'],
            content = request.POST['content'],
            # regdate = datetime.now()
            regdate = timezone.now()
        ).save()
    
    # return HttpResponseRedirect('/guest/select') # 추가 후 목록 보기    
    return redirect('/guest/select') # 추가 후 목록 보기
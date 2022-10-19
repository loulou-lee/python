from django.shortcuts import render, redirect
from myboard.models import BoardTab
from datetime import datetime

# Create your views here.
def replyFunc(request):
    try:
        data = BoardTab.objects.get(id=request.GET.get('id'))
        context = {'data_one':data}
        return render(request, 'rep/reply.html', context)
    except Exception as e:
        print('replyFunc err :', e)
        return render(request, 'error.html')

def replyokFunc(request):
    if request.method == 'POST':
        try:
            repGnum = int(request.POST.get('gnum'))
            repOnum = int(request.POST.get('onum'))
            imsiRec = BoardTab.objects.get(id=request.POST.get('id'))
            oldGnum = imsiRec.gnum
            oldOnum = imsiRec.onum
            
            if oldGnum == repGnum:
                oldOnum = oldOnum + 1
                
            # 답글 저장
            BoardTab(
                name = request.POST.get('name'),
                passwd = request.POST.get('passwd'),
                mail = request.POST.get('mail'),
                title = request.POST.get('title'),
                cont = request.POST.get('cont'),
                bip = request.META['REMOTE_ADDR'],
                bdate = datetime.now(),
                readcnt = 0,
                gnum = repGnum,
                onum = oldOnum,
                nested = int(request.POST.get('nested')) + 1
            ).save()
            
            return redirect('/board/list')
        except Exception as e:
            print('replyOkFunc err :', e)
            return render(request, 'error.html')

from django.shortcuts import render

# Create your views here.
def MainFunc(request):
    msg = "<h1>홈페이지</h1>"
    return render(request, 'main.html', {'msg':msg})


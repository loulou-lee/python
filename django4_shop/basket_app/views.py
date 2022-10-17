from django.shortcuts import render

def mainFunc(request):
    return render(request, 'main.html')

def page1Func(request):
    return render(request, 'page1.html')

def page2Func(request):
    return render(request, 'page1.html')

def cartFunc(request):
    name = request.POST["name"]
    price = request.POST["price"]
    product = {'name':name, 'price':price}
    
    productList = []
    
    if "shop" in request.session: # 서버가 session을 만든다
        productList = request.session['shop'] # 세션 내에 'shop'이라는 키로 productList 등록
        productList.append(product)
        request.session['shop'] = productList
    else:
        productList.append(product)
        request.session['shop'] = productList
        
    print(productList)
    context = {}
    context['products'] = request.session['shop']
    return render(request, 'cart.html', context)

def buyFunc(request):
    if "shop" in request.session:
        productList = request.session['shop']
        total = 0
        
        for p in productList:
            total += int(p['price'])
            
        print('결제 총액 : ', total)
        request.session.clear() # 세션 내의 모든 키 삭제
        # del request.session['shop'] # 특정 키를 가진 세션 내용 삭제
        
    return render(request, "buy.html", {'total':total})
        
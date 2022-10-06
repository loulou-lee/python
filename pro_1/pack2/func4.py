# 함수 : argument 와 parameter 키워드로 matching 하기
# 매개변수 유형
# 위치 매개변수 : 인수와 순서대로 대응
# 기본값 매개변수 : 매개변수에 입력값이 없으면 기본값 사용
# 키워드 매개변수 : 인수와 매개변수를 동일 이름으로 대응
# 가변 매개변수 : 인수의 갯수가 동적인 경우

def showGugu(start, end=5):
    for dan in range(start, end + 1):
        print(str(dan) + '단 출력')
        
showGugu(2, 3)
print()
showGugu(3)
print()
showGugu(start=2, end=3)
print()
showGugu(end=3, start=2)
print()
showGugu(2, end=3)
print()
# showGugu(start=2, 3) # SyntaxError: positional argument follows keyword argument
# 두번째 인수를 상수로 주면 에러
# showGugu(end=3, 2)

print('\n가변인수 처리')
def func1(*ar):
    # print(ar, end = ' ')
    for i in ar:
        print('후식 : ' + i)
    print()

print()
func1('과일','멜론')
func1('과일','멜론','채소')

print()
def func2(a, *ar):
# def func2(*ar, a): # err
    print(a)
    for i in ar:
        print('후식 : ' + i)
    print()
    
func2('과일','멜론')
func2('과일','멜론','채소')

print()
def calcProcess(op, *ar):
    if op == 'sum':
        re = 0
        for i in ar:
            re += i
    elif op == 'mul':
        re = 1
        for i in ar:
            re *= i             
    return re

print(calcProcess('sum', 1,2,3,4,5))
print(calcProcess('mul', 1,2,3,4,5))
print(calcProcess('mul', 1,2,3,4,5,1,2,3,4,5))

print()
def func3(w, h, **other):
    print('w:{}, h:{}'.format(w, h))
    print(other)
    
func3(55, 160)
func3(55, 160, irum='홍길동') # dict 자체로 넘기면 안된다 {'irum':'홍길동'} 
func3(55, 160, irum='홍길동', age=23)

print()
def func4(a, b, *c, **d):
    print(a, b)
    print(c)
    print(d)
    
func4(1, 2)
func4(1, 2, 3) # (3,) -> 튜플형
func4(1, 2, 3, 4, 5)
func4(1, 2, 3, 4, 5, x=6, y=7) 



         
    

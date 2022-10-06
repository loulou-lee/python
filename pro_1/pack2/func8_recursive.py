# 재귀 함수(Recursive function) : 함수가 자기 자신을 호출 - 반복 처리

def CountDown(n):
    if n == 0:
        print('완료')
    else:
        print(n, end = ' ')
        CountDown(n - 1) # <=== 요거
        
CountDown(5)
print()

print('1 ~ 10 까지의 합 구하기')
def tot(n):
    if n == 1:
        print('탈출')
        return True
    return n + tot(n - 1)

result = tot(10)
print('1 ~ 10 까지의 합은 ', result)

print('factorial : 1 부터 어떤 양의 정수 n까지의 정수를 모두 곱한 것 n!')
# 5! = 5 * 4 * 3 * 2 * 1

def factoFunc(a):
    if a == 1:return 1
    print(a)
    return a * factoFunc(a - 1)

result = factoFunc(5)
print('5! : ', result)
print(5 * 4 * 3 * 2 * 1)    

    
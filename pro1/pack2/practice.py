# 자원의 재활용 : 클래스는 다른 클래스를 불러다 사용 가능
# 클래스의 포함관계(has a)

class Machine: #  어떤 클래스에서든 호출될 수 있다.
    cupCount = input('몇 잔을 원하세요 : ') # 몇잔
   
    def showData(self):
        if int(self.cupCount) > 0:
            print('커피 ' + str(self.cupCount) + '잔과 잔돈 ')
            return ''
        elif int(self.cupCount) <= 0:
            print('한잔 이상 입력하세요')
            
class CoinIn:
    coin = input('동전을 입력하세요 : ')
    change = ''
    
    def __init__(self): #생성자
        self.handle = Machine() # 클래스의 포함
        
    def culc(self, cupCount):
        if cupCount > 0:
            cupCount = int(self.handle.cupCount)
            self.change = int(self.coin) - 200*cupCount
            return self.change
            #if self.change < 0:
                #print('')
        elif cupCount <= 0:
            return ''
            
if __name__ == '__main__':
    order = CoinIn()
    #print(order.handle.showData())
    a = int(order.handle.cupCount)
    #print('커피 ' + str(order.handle.cupCount) + '잔과 잔돈 ' + str(order.culc(a)))
    print(order.handle.showData()) #+ str(order.culc(a)))
    print(str(order.culc(a)))
    
'''
# 클래스의 포함관계 연습문제 

class CoinIn:
    def calc(self, cupCount):
        re = ""
        
        if self.coin < 200:
            re = "요금이 부족하네요"
        elif cupCount * 200 > self.coin:
            re = "요금이 부족하네요"
        else:
            self.change = self.coin - (200 * cupCount)  # 잔돈 계산
            re = "커피 {}잔과 잔돈 {}원".format(cupCount, self.change)

        return re

class Machine():
    cupCount = 1  # 현재 코드에서는 의미 없음

    def __init__(self):
        self.coinIn = CoinIn()  # 포함

    def showData(self):
        self.coinIn.coin = int(input("동전을 입력하세요 :"))
        self.cupCount = int(input("몇 잔을 원하세요 :"))

        print(self.coinIn.calc(self.cupCount))


if __name__ == '__main__':
    Machine().showData()
'''
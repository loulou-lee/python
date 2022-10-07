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
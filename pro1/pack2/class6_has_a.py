# 냉장고에 음식 담기 - 클래스의 포함관계로 구현

class Fridge:
    isOpened = False
    foods = []
    
    def open(self):
        self.isOpened = True
        print('냉장고 문 열기')
        
    def put(self, thing):
        if self.isOpened:
            self.foods.append(thing) # 포함
            print('냉장고 안에 음식을 저장함')
            self.food_list()
        else:
            print('냉장고 ㅜㄴ이 닫혀 있어 음식을 저장할 수 없어요')
            
    def close(self):
        self.isOpened = False
        print('냉장고 문 닫기')
        
    def food_list(self):
        for f in self.foods:
            print('-', f.irum, f.expiry_date)
        print()
        
class FoodData:
    def __init__(self, irum, expiry_date):
        self.irum = irum
        self.expiry_date = expiry_date
        
if __name__ == '__main__':
    f = Fridge()
    
    apple = FoodData('사과', '2022-10-15')
    f.put(apple)
    f.open()
    f.put(apple)
    f.close()
    print()
    cider = FoodData('칠성사이다', '2023-10-25')
    f.open()
    f.put(cider)
    f.close()
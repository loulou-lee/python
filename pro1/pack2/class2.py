# 클래스

class Car:
    handle = 0 # Car type의 객체에서 참조 가능 멤버 필드
    speed = 0
    
    def __init__(self, name, speed):
        self.name = name
        self.speed = speed
        
    def showData(self): # Car type의 객체에서 참조 가능 멤버 메소드
        km = '킬로미터'
        msg = '속도: ' + str(self.speed) + km
        return msg
    
print(id(Car))
print(Car.handle)
print(Car.speed)
print()
car1 = Car('tom', 10) # 생성자 호출 후 객체 생성
print(car1.handle, car1.name, car1.speed) # car1 객체에 handle이 없을때 Car의 handle 값 가져옴
car1.color = '보라'
print('car21 color : %s'%car1.color)
print('---')
car2 = Car('james', 20)
print(car2.handle, car2.name, car2.speed)
print('car2 color : %s'%car2.color) # 'Car' object has no at 
 
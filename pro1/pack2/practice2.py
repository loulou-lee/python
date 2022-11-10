# https://dojang.io/mod/page/view.php?id=2389
from abc import * # 추상 클래스를 만들려면 import로 abc 모듈을 가져와야 함

class Employee(metaclass=ABCMeta): # 클래스의 ( )(괄호) 안에 metaclass=ABCMeta를 지정
    
    name = ''
    age = ''
    
    @abstractmethod #@abstractmethod를 붙여서 추상 메서드로 지정
    def pay(self):
        pass
    
    @abstractmethod
    def data_print(self):
        pass
    
    def info_print(self):
        print(self.name+str(self.age))
        
class Regular(Employee):
    
    salary = 0
    
    def __init__(self,name,age,salary):
        self.name
        self.age
        self.salary
    
    def data_print(self):
        print('이름:'+self.name+'나이:'+self.age+'급여:'+self.salary)
    
class Temporary(Employee):
    
    day = 0
    daily_pay = 0
    
    def __init__(self,name,age,day,daily_pay):
        self.name
        self.age
        self.day
        self.daily_pay
    
    def data_print(self):
        print('이름:'+self.name+'나이:'+self.age+'월급:'+self.day*self.daily_pay)
    
        
class Salesman():
    
    sales = 0
    commission = 0
    
    def __init__(self,name,age,salary,sales,commission):
        self.name
        self.age
        self.salary
        self.sales
        self.commission   
    
'''
class Employee(metaclass = ABCMeta):
    def __init__(self, irum, nai):
        self.irum = irum
        self.nai = nai

    @abstractmethod
    def pay(self):
        pass

    @abstractmethod
    def data_print(self):
        pass

    def irumnai_print(self):
        print('이름 : ' + self.irum + ', 나이 : ' + str(self.nai), end=' ')


class Temporary(Employee):
    def __init__(self, irum, nai, ilsu, ildang):
        Employee.__init__(self, irum, nai)   
        self.ilsu = ilsu;
        self.ildang = ildang;

    def pay(self):
        return self.ilsu * self.ildang

    def data_print(self):
        self.irumnai_print();
        print(', 월급 : ' + str(self.pay()))
        
class Regular(Employee):
    def __init__(self, irum, nai, salary):
        super().__init__(irum, nai)
        self.salary = salary

    def pay(self):
        return self.salary

    def data_print(self):
        self.irumnai_print();
        print(', 급여 : ' + str(self.pay()))
        
class Salesman(Regular):
    def __init__(self, irum, nai, salary, sales, commission):
        super().__init__(irum, nai, salary)
        self.sales = sales;
        self.commission = commission
        
    def pay(self):
        return super().pay() + (self.sales * self.commission)

    def data_print(self):
        self.irumnai_print();
        print(', 수령액 : ' + str(round(self.pay())))


t = Temporary('홍길동', 25, 20, 150000);
r = Regular('한국인', 27, 3500000)
s = Salesman('손오공', 29, 1200000, 5000000, 0.25)
t.data_print();   r.data_print();  s.data_print()
'''
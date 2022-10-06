# Module : 소스 코드의 재사용을 가능하게 할 수 있으며, 소스코드를 하나의 이름 공간으로 구분하고 관리하게 된다.
# 하나의 파일은 하나의 모듈이 된다.
# 표준모듈, 사용자 작성 모듈, 제 3자(Third party) 모듈

# 모듈의 멤버 : 전역변수, 실행문, 함수, 클래스, 모듈

a = 10
print(a)

def abc():
    print('abc는 모듈의 멤버 중 함수')

abc()

# 표준모듈(내장된 모듈 읽기)
import math
print(math.pi)
print(math.sin(math.radians(30)))

import calendar
calendar.setfirstweekday(6) # 0:월...일(6)
calendar.prmonth(2022, 10)

import os
print(os.getcwd())
print(os.listdir('/'))

print()
import random
print(random.random())
print(random.randint(1, 10))

from random import random
print(random())

from random import randint, randrange
print(randint(1, 10))

from random import * # import java.sql.* -> 전부 램으로 로딩

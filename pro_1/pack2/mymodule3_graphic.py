# 파이썬 제공 그래픽 라이브러리(모듈들의 집합)
# turtle로 도형 그리기
from turtle import *

p = Pen()

p.color('red', 'yellow')
p.begin_fill()
while True:
    forward(200)
    left(170)
    if abs(pos()) < 1:
        break
p.end_fill()
done()



# classification : 주어진 feature에 대해 label로 학습시켜 데이터를 분류하는 방법
# h0(x) = P(y=1|x;0) x:feature, 0:model parameter
# hypothesis function의 출력값은 "주어진 feature x라는 값을 가질 때 class 1에 들어갈 확률"이라는 의미
# P(y=0|x;0) + P(y=1|x;0)=1

print('Functional api 사용 -----')
from keras.models import Model
from keras.layers import Input


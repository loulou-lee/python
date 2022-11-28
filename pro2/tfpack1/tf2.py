# 변수 : 모델 학습 시, 매개변수 갱신 등을 위해 사용

import tensorflow as tf
from _pytest import outcomes

f = tf.Variable(1.0)
v = tf.Variable(tf.ones((2,)))
m = tf.Variable(tf.ones((2, 1)))
print(f, v)
print(m)
print(m.numpy())

print()
v1 = tf.Variable(1) # 0-d tensor
print(v1)
#v1 = 77 # AttributeError: 'int' object has no attribute 'assign'

v1.assign(10) # 값 치환
print(v1, v1.numpy(), type(v1))

v2 = tf.Variable(tf.ones(shape=(1))) # 1-d tensor
v2.assign([20])
print(v2, v2.numpy(), type(v1))

v3 = tf.Variable(tf.ones(shape=(1, 2))) # 2-d tensor
v3.assign([[20, 30]])
print(v3, v3.numpy(), type(v3))

print()
v1 = tf.Variable([3])
v2 = tf.Variable([5])
v3 = v1 * v2 + 10
print(v3)

print()
var = tf.Variable([1,2,3,4,5], dtype=tf.float32)
result1 = var + 10
print(result1)

print()
w = tf.Variable(tf.ones(shapo=(1,)))
b = tf.Variable(tf.ones(shapo=(1,)))
w.assign([2])
b.assign([3])

def func1(x):
    return w*x + b

out_a1 = func1([[3]])
print('out_a1 : ', out_a1)

print()
@tf.function # auto graph 기능이 적용된 함수 : tf.Graph + tf.Session이 적용
def func2(x):
    return w * x + b

print(type(func2))
out_a2 = func2([1, 2])
print(out_a2)

print()
rand = tf.random.uniform([5], 0, 1) # 균등분포
print(rand.numpy()) 
rand2 = tf.random.normal([5], mean=0, stddev=1) # 정규분포
print(rand2.numpy())

print()
aa = tf.ones((2, 1))
print(aa.numpy())

m = tf.Variable(tf.zeros((2,1)))
print(m.numpy())
m.assign(aa)
print(m.numpy())

m.assign_add(aa)
print(m.numpy())

m.assign_sub(aa)
print(m.numpy())

print('---TF의 구조 (Graph로 설계된 내용은 Session에 실행)')
g1 = tf.Graph()

with g1.as_default(): # 특정 자원 처리를 한 후 자동 close()
    c1 = tf.constant(1, name='c_one')
    c1_1 = tf.constant(1, name='c_one_1')
    print(c1)
    print(type(c1)) # Tensor 객체
    
    print(c1.op) # tf.Operation 객체
    print(g1.as_graph_def())
    
g2 = tf.Graph()
with g2.as_default():
    v1 = tf.Variable(initial_value=1, name='v1')
    print(v1)
    print(type(v1))
    print(v1.op)
    
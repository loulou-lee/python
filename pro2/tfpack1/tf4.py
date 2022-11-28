# 연산자와 기본 함수 경험
import tensorflow as tf
import numpy as np

x = tf.constant(7)
y = tf.constant(3)

print(x + y)
print(tf.add(x, y))

print(tf.cond(x > y, lambda:tf.add(x, y), lambda:tf.subtract(x, y)))

f1 = lambda:tf.constant(123)
f2 = lambda:tf.constant(456)

imsi = tf.case([(tf.greater(x, y), f1)], default=f2) # tf.less
print(imsi.numpy())

print('관계 연산')
print(tf.equal(1, 2).numpy())
print(tf.not_equal(1, 2).numpy())
print(tf.greater(1, 2).numpy())
print(tf.less(1, 2).numpy())

print('논리 연산')
print(tf.logical_and(True, False).numpy())
print(tf.logical_or(True, False).numpy())
print(tf.logical_not(True).numpy())

#####

print()
# tf.reduce~ : 차원 축소
ar = [[1,2],[3,4]]

t = np.array([[[0,1,2],[3,4,5]],[[6,7,8],[9,10,11]]])
print(t.shape)

# 차원 축소
aa = np.array([[1],[2],[3],[4]])
print(aa.shape)
bb = tf.squeeze(aa)
print(bb)

print()
# 차원 확대
tarr = tf.constant([[1,2,3],[4,5,6]])
print(tf.shape(tarr))
print(tf.shape(tarr))
sbs = tf.expand_dims(tarr, 0) # 첫번째 차원을 추가해 확장
print(sbs, tf.shape(sbs).numpy())

print(tf.one_hot)
print(tf.one_hot())


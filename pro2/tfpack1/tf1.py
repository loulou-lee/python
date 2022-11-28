import tensorflow as tf
import os
# SSE 및 AVX 등의 경고는 소스를 빌드 하면 없어지지만, 명시적으로 경고 없애기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("즉시 실행 모드: ", tf.executing_eagerly())
print("GPU ", "사용 가능" if tf.test.is_gpu_available() else "사용 불가능")
print(tf.__version__) # 2.11.0


# 상수 정의(생성) - 상수 텐서를 생성한다.
# print(tf.constant(1)) # tf.Tensor(1, shape=(), dtype=int32)
# print(tf.constant([1])) # tf.Tensor([1], shape=(1,), dtype=int32)
# print(tf.constant([[1]])) # tf.Tensor([[1]], shape=(1, 1), dtype=int32)

# 텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조
# 텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있다는 점만 제외하면 NumPy 의 ndarray와 유사하다

# 상수 선언
print(1, type(1))
print(tf.constant(1), type(tf.constant(1))) # scala : 0-d tensor
print(tf.constant([1])) # vector : 1-d tensor 1차원 배열
print(tf.constant([[1]])) # matrix : 2-d tensor 2차원 배열
print(tf.rank(tf.constant(1)), ' ', tf.rank(tf.constant([1])), ' ', tf.rank(tf.constant([[1]])))

print()
a = tf.constant([1, 2])
b = tf.constant([3, 4])
c = a + b
print(c) # tf.Tensor([4 6], shape=(2,), dtype=int32)
c = tf.add(a, b)
print(c)

print()
# d = tf.constant([3]) # Broadcasting
d = tf.constant([[3]]) # Broadcasting
e = c + d
print(e)

print(1 + 2)
print(tf.constant([1]) + tf.constant([2])) # 연산하는 동네가 다르다

print()
print(7)
print(tf.convert_to_tensor(7, dtype=tf.float32))
print(tf.cast(7, dtype=tf.float32))
print(tf.constant(7.0))
print(tf.constant(7, dtype=tf.float32))

# graph가 병렬연산을 해서 GPU 필요
print()
# numpy의 ndarray와 tensor 사이에 type 변환
import numpy as np
arr = np.array([1, 2])
print(arr, type(arr))
print(arr + 5)

tfarr = tf.add(arr, 5) # ndarray가 자동으로 tensor로 변환
print(tfarr)
print(tfarr.numpy()) # numpy type으로 강제 형변환
print(np.add(tfarr, 3)) # numpy type으로 자동 행변환
print(list(tfarr.numpy()))









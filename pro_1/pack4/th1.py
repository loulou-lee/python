# process : 실행 중인 프로그램을 의미함. 자신만의 메모리를 확보하고 공유하지 않는다.
# thread : light weight process라고도 함. 하나의 process 내에는 한 개의 thread가 존재함
# process 내에 여러개의 thread를 운영하여 여러개의 작업을 동시에 하는 것처럼 느끼게 할 수 있다.
# multi thread로 multi tasking이 가능

import threading, time

def run(id):
    for i in range(1, 51):
        print('id:{} --> {}'.format(id, i))
        time.sleep(0.2)
        
# thread를 사용하지 않은 경우
# run(1)
# run(2)

# thread를 사용한 경우
# threading.Thread(target=수행함수명)
th1 = threading.Thread(target=run, args=('일',))
th2 = threading.Thread(target=run, args=('이',))
th1.start()
th2.start()
th1.join() #메인 스레드가 대기 상태  해당 쓰레드가 종료할 때까지 대기합니다?
th2.join()

print('프로그램 종료')
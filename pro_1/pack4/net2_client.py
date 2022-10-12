# 단순 클라이언트
from socket import *

clientsocket = socket(AF_INET, SOCK_STREAM)
clientsocket.connect(('192.168.0.75', 7878)) # 능동적으로 server에 접속
clientsocket.send('les안녕 반가워'.encode(encoding='utf_8'))
re_msg = clientsocket.recv(1024).decode() # 수신
print('수신자료 : ', re_msg)
clientsocket.close()

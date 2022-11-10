# 네트워킹 프로그래밍
# TCP protocol 기반의 socket(네트워크를 위한 통신채널 지원 클래스 또는 함수)

import socket

print(socket.getservbyname('http', 'tcp'))
print(socket.getservbyname('telnet', 'tcp'))
print(socket.getservbyname('ftp', 'tcp'))
print(socket.getservbyname('SMTP', 'tcp'))
print(socket.getservbyname('pop3', 'tcp'))

print(socket.getaddrinfo('www.naver.com', 80, proto=socket.SOL_TCP))
# '223.130.195.95' '223.130.195.200' 
# 여러 주소 중에 랜덤으로 2개를 getaddrinfo 메소드가 반환한다 (네이버에서 두개만 준다)
# http://223.130.200.107:80/index.html



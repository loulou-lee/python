# 웹 서비스 대상 파일
kor = 50
eng = 60
tot = kor + eng

print('Content-Type:text/html;charset=utf-8\n') # 브라우저로 전송한다
print('<html><body>')
print('<b>안녕하세요</b> 파이썬 모듈로 작성했어요<br>')
print('총점은 %s'%(tot))
print('</body></html>')
# 원격 데이터베이스 연동 프로그램
# pip install mysqlclient
import MySQLdb

# conn = MySQLdb.connect(host = '127.0.0.1', user = 'root', password='123', database='test')
# print(conn)
# conn.close()

# sangdata table과 연동
config = {

    'host':'127.0.0.1',

    'user':'root',

    'password':'123',

    'database':'test',

    'port':3306,

    'charset':'utf8',

    'use_unicode':True

}

try:
    conn = MySQLdb.connect(**config) # **은 dict를 받는 것
    # print(conn)
    cursor = conn.cursor()
    
    '''
    # insert
    # sql = "insert into sangdata(code, sang, su, dan) values(10, '신상1',5,5000)"
    sql = "insert into sangdata values(%s,%s,%s,%s)"
    sql_data = '11','아아',12,5500 #()넣어도 된다
    count = cursor.execute(sql, sql_data)
    print(count)
    conn.commit()
    
    # cursor.execute(sql)
    cursor.execute(sql, sql_data)
    conn.commit() # 자동 커밋 안돼서 커밋해야된다
    '''
    
    """
    # insert
    sql = "update sangdata set sang=%s,su%s where code=%s"
    sql_data = ('파이썬',50,11)
    count = cursor.execute(sql, sql_data)
    print(count)
    conn.commit()
    """
    
    '''
    # delete
    code = '10'
    # sql = "delete from sangdata where code=" + code # secure coding 가이드에 위배
    # sql = "delete from sangdata where code='{0}'".format(code)
    # cursor.execute(sql)
    sql = "delete from sangdata where code=%s"
    cursor.execute(sql, (code,))
    conn.commit()
    '''
    
    # select
    sql = "select code as 코드, sang, su, dan from sangdata"
    cursor.execute(sql)
    
    # 방법1
    for data in cursor.fetchall():
        # print(data)
        print('%s %s %s %s'%data)
        
    # 방법2
    print()
    for r in cursor:
        # print(r)
        print(r[0],r[1],r[2],r[3])
        
    # 방법3
    print()
    for (code,sang,su,dan) in cursor:
        print(code,sang,su,dan)
        
    # 방법3-1
    print()
    for (a,품명,su,kbs) in cursor:
        print(code,sang,su,dan)
    
except Exception as e:
    print('err :', e)
finally:
    cursor.close()
    conn.close()

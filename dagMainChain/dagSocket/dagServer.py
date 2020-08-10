import socket
import threading
import time
import sys
import struct
import os
import shutil
from dagComps import transaction
import json

def socket_service(local_addr, dag_pool, beta):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 6666))
        s.listen(5)
        print('DAG_socket starts...')
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    while 1:
        conn, addr = s.accept()
        t = threading.Thread(target=deal_data, args=(conn, addr, dag_pool, beta))
        t.start()

def send_file(conn,file_addr):
    if os.path.isfile(file_addr):
        fileinfo_size = struct.calcsize('128si')
        fhead = struct.pack('128si', os.path.basename(file_addr).encode('utf-8'), os.stat(file_addr).st_size)
        conn.send(fhead)
        fp = open(file_addr, 'rb')
        response = conn.recv(1024)
        while 1:
            data = fp.read(1024)
            if not data:
                print('{0} file send over...'.format(os.path.basename(file_addr)))
                break
            conn.send(data)

def deal_data(conn, addr, dag_pool, beta):
    print('Accept new connection from {0}'.format(addr))
    conn.send(("You've connected, wait for command...").encode())
    while 1:
        data = conn.recv(1024)
        msg = data.decode()
        if msg == 'exit' or msg == 'require' or msg == 'requireTips' or msg == 'transUpload' or msg == 'delay':
            print('{0} client send data is {1}'.format(addr, msg))
        time.sleep(1)
        if msg == 'exit' or not data:
            print('{0} connection close'.format(addr))
            conn.send('Connection closed!'.encode())
            break
        ### deduce the data type
        if msg == 'require':
            conn.send('ok'.encode())
            data_t = conn.recv(1024)
            msg_t = data_t.decode()
            require_trans_file = './dagSS/dagPool/'+msg_t+'.json'
            send_file(conn, require_trans_file)
        elif msg == 'requireTips':
            conn.send('ok'.encode())
            data_t = conn.recv(1024)
            msg_t = data_t.decode()
            require_tips_file = './dagSS/'+msg_t+'.json'
            send_file(conn, require_tips_file)
        elif msg == 'transUpload':
            conn.send('ok'.encode())
            data_t = conn.recv(1024)
            conn.send('ok'.encode())
            msg_t = json.loads(data_t.decode("utf-8"))
            new_trans = transaction.Transaction(**msg_t)
            transaction.save_transaction(new_trans, './dagSS/dagPool/')
            dag_pool.DAG_publish(new_trans, beta)
            transName = 'node{}_'.format(new_trans.src_node) + str(new_trans.timestamp)
            print('*******************************************')
            print('The new trans *'+transName+'* had been published!')
            print('*******************************************')
        elif msg == 'delay':
            conn.send(str(time.time()).encode())
    conn.close()

# if __name__ == '__main__':
#     socket_service(sys.argv[1])
import socket
import sys
import struct
import os
import time
import json

def socket_client(aim_addr):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((aim_addr, 6666))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print(s.recv(1024))
    while 1:
        data = input('please input work: ').encode()
        s.send(data)
        print('aa', s.recv(1024))
        if data == 'exit':
            break
    s.close()

def back_delay(aim_addr):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((aim_addr, 6666))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    client_time = time.time()
    print(s.recv(1024).decode())
    data = 'delay'.encode()
    s.send(data)
    sever_time = float(s.recv(1024).decode())
    delta = sever_time - client_time
    return delta

def client_trans_require(aim_addr,trans_name,DAG_file):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((aim_addr, 6666))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print(s.recv(1024).decode())
    data = 'require'.encode()
    s.send(data)
    response = s.recv(1024)
    data = trans_name.encode()
    s.send(data)
    rev_file(s,DAG_file)
    s.close()

def client_tips_require(aim_addr,tips_list,tips_file):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((aim_addr, 6666))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print(s.recv(1024).decode())
    data = 'requireTips'.encode()
    s.send(data)
    response = s.recv(1024)
    data = tips_list.encode()
    s.send(data)
    rev_file(s,tips_file)
    s.close()

def trans_upload(aim_addr,trans_info):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((aim_addr, 6666))
    except socket.error as msg:
        print(msg)
        sys.exit(1)
    print(s.recv(1024).decode())
    data = 'transUpload'.encode()
    s.send(data)
    response = s.recv(1024)
    data = json.dumps(trans_info.json_output()).encode("utf-8")
    s.send(data)
    response = s.recv(1024)
    s.close()

def rev_file(conn,file_addr):
    while 1:
        fileinfo_size = struct.calcsize('128si')
        buf = conn.recv(fileinfo_size)
        if buf:
            filename, filesize = struct.unpack('128si', buf)
            fn = filename.strip(b'\00')
            fn = fn.decode()
            print('File name is {0}, filesize is {1}'.format(str(fn), filesize))
            recvd_size = 0
            fp = open(file_addr, 'wb')
            print('Start receiving...')
            conn.send('OK'.encode())
            while not recvd_size == filesize:
                if filesize - recvd_size > 1024:
                    data = conn.recv(1024)
                    recvd_size += len(data)
                else:
                    data = conn.recv(filesize - recvd_size)
                    recvd_size = filesize
                fp.write(data)
            fp.close()
            print('End receive...')
        break

if __name__ == '__main__':
    socket_client(sys.argv[1])
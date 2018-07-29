#!/usr/bin/python3  
#-*- coding: utf-8 -*-  
import sys  
import os  
import threading  
import socket  
from urllib import request  
  
timeout = 4  
socket.setdefaulttimeout(timeout)  
  

def download_and_save(url,savename):  
    try:
        headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}  
        req = request.Request(url, headers=headers)    
        data = request.urlopen(req).read()  
        image_file = open(savename,'w+b')  
        image_file.write(data)  
        print("download succeed: "+ url)  
        image_file.close()  
    except Exception as e:  
        print("download failed: "+ url)
        # print(e.message) 
  
  
def get_all_iamge(filename,train_dir,test_dir):  
    image_file = open(filename)  
    name = filename.split('/')[-1]  
    name = name[:-4]  
    lines = image_file.readlines()  
    for line in lines:  
        line_split = line.split(' ')  
        image_id = line_split[0]  
        image_url = line_split[1]
        if  int(image_id) < 700:
            if not os.path.exists(train_dir + '/' + name):  
                os.makedirs(train_dir + '/' + name)  
            savefile = 'train_dir' + '/' + name + '/' + image_id + '.jpg'
        else:
            if not os.path.exists(test_dir + '/' + name):  
                os.makedirs(test_dir + '/' + name)  
            savefile = 'test_dir' + '/' + name + '/' + image_id + '.jpg'
        #The maxSize of Thread numberr:1000  
        # print(image_url,savefile)  
        while True:  
            if(len(threading.enumerate()) < 1000):  
                break                 
        t = threading.Thread(target=download_and_save,args=(image_url,savefile,))  
        t.start()  
  
if __name__ == "__main__":   
    file_dir = sys.argv[1]
    dataset_dir = sys.argv[2]
    train_dir = os.path.join(dataset_dir,'train')
    test_dir = os.path.join(dataset_dir,'test')
    list = os.listdir(file_dir)  
    for i in range(len(list)):  
        get_all_iamge(os.path.join(file_dir,list[i]),train_dir,test_dir)  

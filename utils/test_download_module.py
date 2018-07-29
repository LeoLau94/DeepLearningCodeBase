import sys  
import os  
import threading  
import socket  
from urllib import request  

if __name__ == '__main__':
	url = 'http://cdn02.cdn.socialitelife.com/wp-content/uploads/2011/07/ted-danson-santa-monica-90210-07132011-04-435x580.jpg'
	headers = {'User-Agent': 'User-Agent:Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36'}  
	req = request.Request(url, headers=headers)
	data = request.urlopen(req).read()
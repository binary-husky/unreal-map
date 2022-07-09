import platform
from sys import stdout

if platform.system()=="Linux":
    pass
else: 
    from colorama import init
    init()

# Do you like the elegance of Chinese characters?
def print红(*kw,**kargs):
    print("\033[0;31m",*kw,"\033[0m",**kargs)
def print绿(*kw,**kargs):
    print("\033[0;32m",*kw,"\033[0m",**kargs)
def print黄(*kw,**kargs):
    print("\033[0;33m",*kw,"\033[0m",**kargs)
def print蓝(*kw,**kargs):
    print("\033[0;34m",*kw,"\033[0m",**kargs)
def print紫(*kw,**kargs):
    print("\033[0;35m",*kw,"\033[0m",**kargs)
def print靛(*kw,**kargs):
    print("\033[0;36m",*kw,"\033[0m",**kargs)

def print亮红(*kw,**kargs):
    print("\033[1;31m",*kw,"\033[0m",**kargs)
def print亮绿(*kw,**kargs):
    print("\033[1;32m",*kw,"\033[0m",**kargs)
def print亮黄(*kw,**kargs):
    print("\033[1;33m",*kw,"\033[0m",**kargs)
def print亮蓝(*kw,**kargs):
    print("\033[1;34m",*kw,"\033[0m",**kargs)
def print亮紫(*kw,**kargs):
    print("\033[1;35m",*kw,"\033[0m",**kargs)
def print亮靛(*kw,**kargs):
    print("\033[1;36m",*kw,"\033[0m",**kargs)

if not stdout.isatty():
    # redirection, avoid a fucked up log file
    print红 = print
    print绿 = print
    print黄 = print
    print蓝 = print
    print紫 = print
    print靛 = print
    print亮红 = print
    print亮绿 = print
    print亮黄 = print
    print亮蓝 = print
    print亮紫 = print
    print亮靛 = print

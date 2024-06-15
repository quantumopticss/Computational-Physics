## *args 是打包和解包
# 在传递args的时候不加*
# 例如integral接受调用处的 args = (a,b,...)
# 传递给 integral_calculate(...,args,...)
# integral_calculate 调用 func 需要解包，fun(...,*args,...)

def main():
    a = 1
    b = 2
    fun1(args = (a,b,))

def fun1(args = ()):
    x = 3
    new_arg = (x,*args)
    fun2(new_arg)

def fun2(args  =()):
    print(*args)


# 调用 process 函数并传递任意数量的参数
a = [1,2,3]
b = 2*a
print(b)
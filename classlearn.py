#学习类的继承
class Parent:
    def __init__(self):
        print("这是父类的构造函数")

class Child(Parent):
    def __init__(self):
        super().__init__()
        print("这是子类的构造函数")

c = Child()

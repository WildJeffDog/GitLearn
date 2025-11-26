#I just want to be happy
def find_happiness():
    print("输入你能想到的所有事情吧！无论是开心还是烦恼，都可以尽情说出来！\n")
    print("当你觉得说完了，输入'hide'结束对话。\n")
    factors = []
    while True:
        factors.append(input("说说嘛，没事的："))
        if factors[-1].lower() == 'hide':
            factors.pop()  # Remove the 'end' entry
            break
    
if __name__ == "__main__":
    find_happiness()
    print("\n谢谢你分享这些！\n"
        "有道是：\n"
        "抽刀断水水更流，\n举杯消愁愁更愁。\n人生在世不称意，\n明朝散发弄扁舟。\n")




    
import sys
import time

def main():
    # 读取一行输入
    for line in sys.stdin:
        # 模拟处理
        time.sleep(3)
        # 输出阶段1
        print("初步判定为肝脏疾病", flush=True)
        time.sleep(3)
        # 输出最终结果
        print("结果判定为肝功能损伤", flush=True)
        break

if __name__ == '__main__':
    main()
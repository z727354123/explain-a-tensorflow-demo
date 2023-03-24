import os
import pandas as pd
import torch


if __name__ == '__main__':
    if True:
        os.makedirs(os.path.join('..', 'data'), exist_ok=True)
        data_file = os.path.join('..', 'data', 'house_tiny.csv')
        with open(data_file, 'w') as f:
            f.write('NumRooms,AlleyD,Price\n') # 列名
            f.write('NA,Pave,127500\n') # 每⾏表⽰⼀个数据样本
            f.write('2,NA,106000\n')
            f.write('4,NA,178100\n')
            f.write('NA,NA,140000\n')

    if True:
        print("-------------------读取文件----------------------")
        data_file = os.path.join('..', 'data', 'house_tiny.csv')
        data = pd.read_csv(data_file)
        print(data)

    if True:
        print("-------------------处理缺失的值----------------------")
        data_file = os.path.join('..', 'data', 'house_tiny.csv')
        data = pd.read_csv(data_file)
        inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
        print(inputs)
        inputs = inputs.fillna(inputs.mean(numeric_only=True))
        print(inputs)
        print("-------------------处理缺失的值1----------------------")
        print(outputs)
        print("-------------------处理缺失的值2----------------------")
        print(data)
        print("-------------------处理类别----------------------")
        inputs.loc[1, 'AlleyD'] = 'JJ'
        print(inputs)
        inputs = pd.get_dummies(inputs, dummy_na=True)
        print(inputs)
        print("-------------------转换为张量格式----------------------")
        print(inputs.values)
        x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
        print(x)
        print(y)





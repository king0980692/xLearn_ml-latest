import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

if len(sys.argv) < 2:
    print('no argument')
    sys.exit()

data = argv[1]
    
df = pd.read_csv(data)
df.sort_values(by="timestamp", inplace=True)
train_data, test_data = train_test_split(df, train_size=0.8, shuffle=False)


train_data_path = "./sub_data/ml_train.csv"
test_data_path = "./sub_data/ml_test.csv"

train_data.to_csv(train_data_path, index=False, header=False)
test_data.to_csv(test_data_path,index=False, header=False)
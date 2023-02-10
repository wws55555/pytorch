import torch
import numpy as np
import pandas as pd
import torchvision.transforms as tr
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from torch.nn.modules.loss import MSELoss
from google.colab import drive

# 텐서화
class TensorData(Dataset):
  def __init__(self, x_data, y_data):
    self.x_data = torch.FloatTensor(x_data)
    self.y_data = torch.FloatTensor(y_data)
    self.len = self.y_data.shape[0]

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.len

# 맵핑, 드롭아웃
class Regressor(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(13, 50, bias=True)
    self.fc2 = nn.Linear(50, 30, bias=True)
    self.fc3 = nn.Linear(30, 1, bias=True)
    self.dropout = nn.Dropout(0.2)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = self.dropout(F.relu(self.fc2(x)))
    x = F.relu(self.fc3(x))
    return x

# 평가
def evaluate(dataloader):
  predictions = torch.tensor([], dtype=torch.float)
  actual = torch.tensor([], dtype=torch.float)
  with torch.no_grad():
    model.eval()
    for data in dataloader:
      input, label = data
      output = model(input)
      predictions = torch.cat((predictions, output), 0)
      actual = torch.cat((actual, label), 0)
  rmse = np.sqrt(mean_squared_error(predictions, actual))
  return rmse

# 데이터 가져오기
drive.mount('/content/gdrive')
cd/content/gdrive/My Drive/deeplearningbro/pytorch
df = pd.read_csv('./data/reg.csv', index_col= [0])

# 데이터, 라벨 분리
X = df.drop('Price', axis=1).to_numpy()
Y = df['Price'].to_numpy().reshape((-1, 1))

# 학습, 테스트 데이터셋 분리
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5)
train_set = TensorData(X_train, Y_train)
test_set = TensorData(X_test, Y_test)

# cross-validation 작업
kfold = KFold(n_splits=3, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kfold.split(train_set)):

  # 학습, 교차검증 데이터셋 분리
  train_sample = torch.utils.data.SubsetRandomSampler(train_idx)
  val_sample = torch.utils.data.SubsetRandomSampler(val_idx)
  
  # 데이터셋의 배치화
  train_loader = DataLoader(train_set, batch_size=32, sampler=train_sample)
  val_loader = DataLoader(train_set, batch_size=32, sampler=val_sample)
  
  # 모델, loss function, 옵티마이저 설정
  model = Regressor()
  criterion = MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-7)

  # gradient-descending 작업
  loss_ = []
  for epoch in range(400):
    for data in train_loader:
      input, label = data
      optimizer.zero_grad()
      output = model(input)  
      loss = criterion(output, label)
      loss.backward()
      optimizer.step()

  # 교차검증 결과 출력  
  val_rmse = evaluate(val_loader)
  loss_.append(val_rmse)
  print(f"{fold}의 val_rmse는 {val_rmse}입니다")

# loss의 평균값 출력
loss_ = np.array(loss_)
score = np.mean(loss_)
print(score)

# 테스트데이터로 평가
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
test_rmse = evaluate(test_loader)
print(test_rmse)
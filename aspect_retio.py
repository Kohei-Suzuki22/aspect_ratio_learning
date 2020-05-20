import numpy as np
import math

np.random.seed(0)   # 乱数の種を指定して再現性を持たせる
N = 1000


TX = (np.random.rand(N,2) * 1000).astype(np.int32) + 1            # ex)[[1,2],[3,4],・・・[]]  → shape:(1000,2)
TY = (TX.min(axis=1) / TX.max(axis=1) <= 0.2).astype(np.int)[np.newaxis].T    # ex) [[0],[1],[1],[0]] → shape: (1000,1)



MU = TX.mean(axis=0)        # 列ごとの平均  → shape: (2,)  ex) [1,2]
SIGMA = TX.std(axis=0)      # 列ごとの標準偏差  → shape: (2,) ex) [233.2333,342.11]


# 学習データの標準化.   → 平均を0,分散を1にする。
def standardize(x):
  return (x - MU) / SIGMA


TX = standardize(TX)
# ! TYはstandardizeしなくていい。

# ニューラルネットワークの構造

## 全結合
## 入力x: 2
## 隠れ層: 2層
## 出力:  1


### 重みとバイアス

W1 = np.random.randn(2,2)     # randn: 標準正規分布で乱数を生成。 ←→ rand: 0~1の一様変数で乱数を生成
W2 = np.random.randn(2,2)
W3 = np.random.randn(1,2)

b1 = np.random.randn(2)
b2 = np.random.randn(2)
b3 = np.random.randn(1)


# 活性化関数: シグモイド関数
def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


# 順伝播
def forward(x0):
  z1 = np.dot(x0,W1.T) + b1
  x1 = sigmoid(z1)
  z2 = np.dot(x1,W2.T) + b2
  x2 = sigmoid(z2)
  z3 = np.dot(x2,W3.T) + b3
  x3 = sigmoid(z3)
  return z1,x1,z2,x2,z3,x3


# シグモイド関数の微分
def d_sigmoid(x):
  return (1.0 - sigmoid(x)) * sigmoid(x)


# 出力層のデルタ
def delta_output(z,y):
  return (sigmoid(z) - y) * d_sigmoid(z)

# 隠れ層のデルタ
def delta_hideen(z,d,w):
  return d_sigmoid(z) * np.dot(d,w)



# 逆伝播
def backward(y,z3,z2,z1):
  d3 = delta_output(z3,y)
  d2 = delta_hideen(z2,d3,W3)
  d1 = delta_hideen(z1,d2,W2)
  return d3,d2,d1

# 学習率
ETA = 0.001


# 目的関数の重みでの微分
def d_weight(d,x):
  return np.dot(d.T,x)

# 目的関数のバイアスでの微分
def d_bias(d):
  return d.sum(axis=0)

# パラメータの更新
def update_parameters(d3,x2,d2,x1,d1,x0):
  global W3,W2,W1,b3,b2,b1
  # global に指定することで、グローバル変数を更新していける。(通常はメソッド内では、ローカル変数の更新はできるが、グローバル変数の更新ができない。)
  W3 = W3 - ETA * d_weight(d3,x2)
  W2 = W2 - ETA * d_weight(d2,x1)
  W1 = W1 - ETA * d_weight(d1,x0)
  b3 = b3 - ETA * d_bias(d3)
  b2 = b2 - ETA * d_bias(d2)
  b1 = b1 - ETA * d_bias(d1)



# 学習
def train(x,y):
  # 順伝播
  z1,x1,z2,x2,z3,x3 = forward(x)

  # 逆伝播
  d3,d2,d1 = backward(y,z3,z2,z1)

  # パラメータの更新(ニューラルネットワークの学習)
  update_parameters(d3,x2,d2,x1,d1,x)


# 繰り返しの回数
EPOCH = 30000

# 予測
def predict(x):
  return forward(x)[-1]


# 目的関数
def E(y,x):
  return 0.5 * ((y - predict(x)) ** 2).sum()


BATCH = 100

for epoch in range(1,EPOCH + 1):
  # ミニバッチ学習用にランダムなインデックスを取得
  random_index = np.random.permutation(len(TX))
  # ミニバッチの数分だけデータを取り出して学習
  for i in range(math.ceil(len(TX) / BATCH)):
    indice = random_index[i*BATCH: (i+1) * BATCH]
    x0 = TX[indice]
    y = TY[indice]
    train(x0,y)

  # ログを残す
  if epoch % 1000 == 0:
    log = '誤差 = {:8.4}({:5d}エポック目)'
    print(log.format(E(TY,TX),epoch))


testX = standardize([
  [100,100],
  [100,10],
  [10,100],
  [80,100]
])


print(predict(testX))

# 分類器
def classify(x):
  return (predict(x) > 0.8).astype(np.int)

print(classify(testX))


# テストデータ生成
test_n = 1000
test_x = (np.random.rand(test_n,2) * 1000).astype(np.int32) + 1
test_y = (test_x.min(axis=1) / test_x.max(axis=1) <= 0.2).astype(np.int)[np.newaxis].T


# 制度計算
accuracy = (classify(standardize(test_x)) == test_y).sum() / test_n
print('精度: {}%'.format(accuracy * 100))

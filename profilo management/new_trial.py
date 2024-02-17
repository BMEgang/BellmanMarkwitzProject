import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import CovarianceShrinkage
from pypfopt import expected_returns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from finrl import config_tickers
import os


# ***************************************** knowledge distillation **************
# 文件夹路径
image_folder = "/Users/ganghu/Desktop/pythonProject1/image"
model_folder = "/Users/ganghu/Desktop/pythonProject1/model"
data_folder = "/Users/ganghu/Desktop/pythonProject1/data"

# 检查并创建图像文件夹
if not os.path.exists(image_folder):
    print(f"make {image_folder} success")
    os.makedirs(image_folder)

# 检查并创建模型文件夹
if not os.path.exists(model_folder):
    print(f"make {model_folder} success")
    os.makedirs(model_folder)

if not os.path.exists(data_folder):
    print(f"make {data_folder} success")
    os.makedirs(data_folder)

# 股票代码
tickers = config_tickers.DOW_30_TICKER
# 尝试下载数据，如果失败则从列表中移除对应的股票代码
successful_tickers = []
for ticker in tickers:
    try:
        data = yf.download(ticker, start="2008-09-01", end="2018-09-30")
        if not data.empty:
            successful_tickers.append(ticker)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

# 数据收集
data = yf.download(successful_tickers, start="2008-09-01", end="2018-09-30")#['Close']
# 初始化一个DataFrame来存储权重
weights_df = pd.DataFrame(index=data.index, columns=[ticker + "_weight" for ticker in successful_tickers])

# 定义历史时间窗口（比如30天）
window = 4 # 4

# 当前月份初始化
current_month = None

# 遍历每个交易日
for i in range(window, len(data)):
    current_date = data.index[i]

    # 检查是否是新的一个月
    if current_date.month != current_month:
        current_month = current_date.month  # 更新当前月份

        # 使用窗口数据
        window_data = data[i - window:i].dropna(axis=1)

        # 计算预期收益率和协方差
        mu = expected_returns.mean_historical_return(window_data)
        # S = risk_models.sample_cov(window_data)
        # 使用CovarianceShrinkage来获得协方差矩阵
        S = CovarianceShrinkage(window_data).ledoit_wolf()

        # 使用马科维茨模型
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

        try:
            weights = ef.max_sharpe(risk_free_rate=0.02)
            cleaned_weights = ef.clean_weights()
        except ValueError as e:
            print(f"无法找到最优夏普比率组合: {e}")
            weights = ef.min_volatility()
            cleaned_weights = ef.clean_weights()

        # 存储权重
        for ticker in successful_tickers:
            if ticker in cleaned_weights:
                weight = cleaned_weights.get(ticker, 0)
                weights_df.loc[data.index[i], ticker + "_weight"] = np.maximum(0, weight)
                # weights_df.loc[data.index[i], ticker + "_weight"] = cleaned_weights[ticker]
            else:
                weights_df.loc[data.index[i], ticker + "_weight"] = 0
    else:
        # 如果不是新的一个月，保持上个月的权重
        weights_df.loc[data.index[i]] = weights_df.loc[data.index[i - 1]]

    # 打印夏普比率，即使未调整权重
    try:
        portfolio_performance = ef.portfolio_performance(risk_free_rate=0.02)
        sharpe_ratio = portfolio_performance[2]
    except ValueError as e:
        sharpe_ratio = "N/A"

    print(f"Date: {current_date.date()}, Sharpe Ratio: {sharpe_ratio}")

# 将权重数据合并到原始数据中
data_combined = pd.concat([data, weights_df], axis=1).fillna(0)
# 将数据集存入csv文件
data_combined.to_csv(f"{data_folder}/all_close_data.csv")

# data_combined 包含历史价格数据和对应的马科维茨模型计算出的权重
# tickers 是股票代码列表
# 生成X和y的列名
X_columns = successful_tickers  # 原始股票价格
y_columns = [ticker + '_weight' for ticker in successful_tickers]  # 对应的权重

# 提取X和y
X = data_combined[X_columns]
y = data_combined[y_columns]

# 根据指定的日期划分数据集
train = data_combined['2009-01-01':'2015-01-01']
test = data_combined['2015-01-01':'2016-01-01']
trade = data['2016-01-01':]

# 分别为训练集和测试集提取特征和标签
X_train = train[X_columns]
y_train = train[y_columns]
X_test = test[X_columns]
y_test = test[y_columns]

# 计算训练集的均值和标准差
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)

# 标准化训练集和测试集
X_train_scaled = (X_train - X_train_mean) / X_train_std
X_test_scaled = (X_test - X_train_mean) / X_train_std

# 转换为 PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled.values.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
X_test_tensor = torch.tensor(X_test_scaled.values.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

# 创建 PyTorch 数据加载器
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一层
        self.fc2 = nn.Linear(128, 64)  # 第二层
        self.fc3 = nn.Linear(64, 64)  # 第三层
        self.fc4 = nn.Linear(64, 32)  # 第四层
        self.fc5 = nn.Linear(32, 32)  # 第五层
        self.fc6 = nn.Linear(32, 16)  # 第六层
        self.fc7 = nn.Linear(16, output_size)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = self.fc7(x)
        return x


# 实例化模型
model = Net(X_train_tensor.shape[1], y_train_tensor.shape[1])

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 200
for epoch in range(epochs):
    total_loss = 0

    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    raw_predictions = model(X_test_tensor)
# 应用非负约束
non_negative_predictions = torch.relu(raw_predictions)

# 应用权重总和约束
summed_predictions = non_negative_predictions.sum(dim=1, keepdim=True)
normalized_predictions = non_negative_predictions / summed_predictions

# 计算测试损失（注意：这里的损失计算可能需要根据新的输出进行调整）
test_loss = criterion(normalized_predictions, y_test_tensor)
print(f'Test Loss: {test_loss.item()}')

# 计算RMSE
rmse = np.sqrt(test_loss.item())
print(f'Test RMSE: {rmse}')

# 可视化预测与实际值
y_pred_np = normalized_predictions.numpy()
y_test_np = y_test_tensor.numpy()

for i in range(y_pred_np.shape[1]):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_np[:, i], label='Actual')
    plt.plot(y_pred_np[:, i], label='Predicted')
    plt.title(f'Prediction vs Actual for {y.columns[i]}')
    plt.xlabel('Time')
    plt.ylabel('Weight')
    plt.legend()
    plt.savefig(f"{image_folder}/plot_{i}.png")  # 保存图像
    plt.close()  # 关闭图像以释放内存

model_save_path = f"{model_folder}/my_pytorch_model.pth"
torch.save(model.state_dict(), model_save_path)

# 预处理trade数据
X_trade = trade[X_columns]
X_trade_scaled = (X_trade - X_train_mean) / X_train_std
X_trade_tensor = torch.tensor(X_trade_scaled.values.astype(np.float32))

# 使用模型进行预测
model.eval()
with torch.no_grad():
    trade_predictions = model(X_trade_tensor)
    # 应用非负约束
    trade_predictions_nn = torch.relu(trade_predictions)
    # 应用权重总和约束
    trade_summed_predictions = trade_predictions_nn.sum(dim=1, keepdim=True)
    trade_normalized_predictions = trade_predictions_nn / trade_summed_predictions

# 保存预测结果
trade_predictions_df = pd.DataFrame(trade_normalized_predictions.numpy(), columns=y_columns, index=trade.index)
trade_predictions_df.to_csv(f"{data_folder}/trade_predictions.csv")

# 初始资金
initial_capital = 10000

# 计算每日股票收益率
stock_returns = trade.pct_change()

# 初始化资金曲线
capital = initial_capital
capital_curve = []

# 遍历交易天数
for date in trade_predictions_df.index:
    if date in stock_returns.index:
        # 获取该日预测权重
        weights = trade_predictions_df.loc[date].values

        # 计算该日投资组合收益率
        daily_return = np.sum(stock_returns.loc[date] * weights)

        # 更新资金总额
        capital *= (1 + daily_return)

    # 记录资金总额
    capital_curve.append(capital)

# 绘制资金曲线
plt.figure(figsize=(12, 6))
plt.plot(trade_predictions_df.index, capital_curve, label='Capital Over Time')
plt.title('Capital Curve from 2016-01-04 to 2018-09-30')
plt.xlabel('Date')
plt.ylabel('Capital in USD')
plt.legend()
plt.savefig(f"{image_folder}/capital_curve.png")
plt.close()

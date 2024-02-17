import random
import yfinance as yf
import numpy as np
import pandas as pd
from finrl import config_tickers
import os
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product
# 获取当前时间并格式化为字符串（例如 '2023-11-04_12-30-00'）
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# ***************************************** knowledge distillation **************
# 基础路径
base_folder = "/Users/ganghu/Desktop/pythonProject1/final"
# 创建带有当前时间的文件夹路径
image_folder = os.path.join(base_folder, f"{current_time}/image")
model_folder = os.path.join(base_folder, f"{current_time}/model")
data_folder = os.path.join(base_folder, f"{current_time}/data")
result_folder = os.path.join(base_folder, f"{current_time}/result")

# 检查并创建图像文件夹
if not os.path.exists(image_folder):
    print(f"make {image_folder} success")
    os.makedirs(image_folder)

# 检查并创建模型文件夹
if not os.path.exists(model_folder):
    print(f"make {model_folder} success")
    os.makedirs(model_folder)

# 检查并创建数据文件夹
if not os.path.exists(data_folder):
    print(f"make {data_folder} success")
    os.makedirs(data_folder)

if not os.path.exists(result_folder):
    print(f"make {result_folder} success")
    os.makedirs(result_folder)

# 股票代码
tickers = config_tickers.DOW_30_TICKER
# 尝试下载数据，如果失败则从列表中移除对应的股票代码
successful_tickers = []
for ticker in tickers:
    try:
        data = yf.download(ticker, start="2008-12-20", end="2018-08-30")
        if not data.empty:
            successful_tickers.append(ticker)
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

# 下载数据
data_combined = yf.download(successful_tickers, start="2008-12-20", end="2018-09-30")["Close"]

# 数据集划分
train = data_combined["2009-01-01":"2015-01-01"]
test = data_combined["2015-01-01":"2016-01-01"]
trade = data_combined["2016-01-01":]


# ONS模型参数初始化
initial_capital = 10000


def create_initial_population(num_stocks, population_size):
    return [np.random.dirichlet(np.ones(num_stocks)) for _ in range(population_size)]


def calculate_fitness(weights, returns):
    # 确保返回正值
    return np.dot(weights, returns) + 1


def select_parents(population, fitness, num_parents):
    # 使用整数索引而不是直接选择父代
    indices = np.random.choice(len(population), size=num_parents, replace=False, p=fitness / fitness.sum())
    parents = [population[index] for index in indices]
    return parents


def crossover(parents, offspring_size):
    offspring = []
    num_parents = len(parents)
    for _ in range(offspring_size):
        # 随机选择两个不同的父代索引
        parent_indices = np.random.choice(num_parents, 2, replace=False)
        p1, p2 = parents[parent_indices[0]], parents[parent_indices[1]]
        # 选择一个基因进行交叉
        gene = random.randint(0, len(p1) - 1)
        child = np.concatenate((p1[:gene], p2[gene:]))
        # 确保权重和为1
        offspring.append(child / np.sum(child))
    return offspring


def mutation(offspring, mutation_rate=0.1):
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            gene = random.randint(0, len(offspring[i]) - 1)
            offspring[i][gene] = np.random.rand()
    return offspring


class Particle:
    def __init__(self, num_stocks):
        self.position = np.random.dirichlet(np.ones(num_stocks))  # 初始化位置
        self.velocity = np.random.rand(num_stocks) - 0.5  # 初始化速度
        self.best_position = np.copy(self.position)
        self.best_fitness = -np.inf

    def update(self, global_best_position, w=0.5, c1=1, c2=2, v_max=0.1):
        # 更新速度
        r1, r2 = np.random.rand(2)
        self.velocity = w * self.velocity + c1 * r1 * (self.best_position - self.position) + c2 * r2 * (
                    global_best_position - self.position)

        # 限制速度
        self.velocity = np.clip(self.velocity, -v_max, v_max)

        # 更新位置
        new_position = self.position + self.velocity
        if np.all(new_position >= 0) and np.sum(new_position) > 0:  # 确保所有权重非负且非零
            self.position = new_position / np.sum(new_position)  # 归一化以保持总和为1
        else:
            self.velocity = -self.velocity  # 如果新位置无效，反转速度


def pso(data, num_particles=30, num_generations=100):
    num_stocks = data.shape[1]
    particles = [Particle(num_stocks) for _ in range(num_particles)]
    global_best_position = np.random.dirichlet(np.ones(num_stocks))
    global_best_fitness = -np.inf

    for _ in range(num_generations):
        returns = data.pct_change().dropna().mean()

        for particle in particles:
            fitness = calculate_fitness(particle.position, returns)
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position

            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position

        for particle in particles:
            particle.update(global_best_position)

    return global_best_position


# 使用粒子群优化近似UP模型
best_weights = pso(train)

# 使用找到的最佳权重模拟交易数据集上的投资
def simulate_investment_with_best_weights(data, initial_capital, best_weights):
    capital = initial_capital
    capital_over_time = []

    for i in range(1, len(data)):
        returns = data.iloc[i] / data.iloc[i - 1] - 1
        portfolio_return = np.dot(returns, best_weights)
        capital *= max(0, 1 + portfolio_return)  # 确保资本不会变成负数
        capital_over_time.append(capital)

    return np.array(capital_over_time)


# 在交易数据集上运行模拟
portfolio_values = simulate_investment_with_best_weights(trade, initial_capital, best_weights)

# 绘制投资组合价值变化
plt.figure(figsize=(12, 6))
plt.plot(trade.index[1:], portfolio_values)
plt.title("Genetic Algorithm Optimized Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.savefig(f"{image_folder}/PSO_Optimized_Portfolio.png")

# 保存投资组合价值到 CSV
portfolio_df = pd.DataFrame({
    'Date': trade.index[1:],
    'Portfolio_Value': portfolio_values
})
csv_save_path = os.path.join(data_folder, "PSO_Optimized_Portfolio.csv")
portfolio_df.to_csv(csv_save_path, index=False)

# 打印粒子群优化算法得到的权重
print("Optimized Weights from PSO:")
print(best_weights)

# 筛选2017年11月至2017年12月的数据
start_date = "2017-11-01"
end_date = "2017-12-01"

# 获取特定时间段的初始和最终资本
initial_capital_period = portfolio_values[trade.index.searchsorted(start_date)]
final_capital_period = portfolio_values[trade.index.searchsorted(end_date) - 1]  # 减去1因为索引是从0开始的

# 计算收益
absolute_return = final_capital_period - initial_capital_period
relative_return = absolute_return / initial_capital_period

# 打印结果
print("\nAbsolute Return from 2017-11 to 2017-12: ${:.2f}".format(absolute_return))
print("Relative Return from 2017-11 to 2017-12: {:.2%}".format(relative_return))

# 计算投资组合的每日收益率
portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]

# 选取特定时间段内的每日收益率
selected_returns = portfolio_returns[(trade.index[1:-1] >= start_date) & (trade.index[1:-1] <= end_date)]

# 计算波动率（标准差）
volatility = np.std(selected_returns)

# 输出风险（波动率）
print("Volatility from 2017-11 to 2017-12: {:.2%}".format(volatility))
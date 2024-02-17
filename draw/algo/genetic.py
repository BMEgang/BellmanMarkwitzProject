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

def genetic_algorithm(data, population_size=50, num_generations=100):
    num_stocks = data.shape[1]
    population = create_initial_population(num_stocks, population_size)

    for generation in range(num_generations):
        returns = data.pct_change().dropna()
        fitness = np.array([calculate_fitness(individual, returns.mean()) for individual in population])
        parents = select_parents(population, fitness, population_size // 2)
        offspring_crossover = crossover(parents, len(population) - len(parents))
        offspring_mutation = mutation(offspring_crossover)
        population = np.vstack((parents, offspring_mutation))

    best_portfolio = population[np.argmax(fitness)]
    return best_portfolio


# 使用遗传算法近似UP模型
best_weights = genetic_algorithm(train)

# 使用找到的最佳权重模拟交易数据集上的投资
def simulate_investment_with_best_weights(data, initial_capital, best_weights):
    capital = initial_capital
    capital_over_time = []

    for i in range(1, len(data)):
        returns = data.iloc[i] / data.iloc[i - 1] - 1
        capital *= (1 + np.dot(returns, best_weights))
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
plt.savefig(f"{image_folder}/Genetic_Optimized_Portfolio.png")

# 保存投资组合价值到 CSV
portfolio_df = pd.DataFrame({
    'Date': trade.index[1:],
    'Portfolio_Value': portfolio_values
})
csv_save_path = os.path.join(data_folder, "Genetic_Optimized_Portfolio.csv")
portfolio_df.to_csv(csv_save_path, index=False)

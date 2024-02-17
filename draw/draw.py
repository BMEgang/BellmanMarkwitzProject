import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pyo
import numpy as np
import seaborn as sns
from math import pi

# 读取数据
knowledgement_drill_ddpg_0 = pd.read_csv("richard.csv", index_col=0,parse_dates=['date'])
knowledgement_drill_ddpg_1 = pd.read_csv("richard1.csv", index_col=0,parse_dates=['date'])
knowledgement_drill_ddpg_2 = pd.read_csv("richard2.csv", index_col=0,parse_dates=['date'])
knowledgement_drill_ddpg_3 = pd.read_csv("richard3.csv", index_col=0,parse_dates=['date'])
knowledgement_drill_ddpg_4 = pd.read_csv("richard4.csv", index_col=0,parse_dates=['date'])

# 合并所有数据帧
knowledgement_drill_ddpgs = knowledgement_drill_ddpg_0
knowledgement_drill_ddpgs["account_value1"] = knowledgement_drill_ddpg_1["account_value"]
knowledgement_drill_ddpgs["account_value2"] = knowledgement_drill_ddpg_2["account_value"]
knowledgement_drill_ddpgs["account_value3"] = knowledgement_drill_ddpg_3["account_value"]
knowledgement_drill_ddpgs["account_value4"] = knowledgement_drill_ddpg_4["account_value"]

# 计算每行的均值和标准差
knowledgement_drill_ddpgs['mean'] = knowledgement_drill_ddpgs[['account_value', 'account_value1', 'account_value2', 'account_value3', 'account_value4']].mean(axis=1)
knowledgement_drill_ddpgs['std'] = knowledgement_drill_ddpgs[['account_value', 'account_value1', 'account_value2', 'account_value3', 'account_value4']].std(axis=1)

# 创建三个跟踪器（trace），一个用于绘制均值，另外两个用于绘制均值加减标准差的范围
trace_mean = go.Scatter(
    x=knowledgement_drill_ddpgs['date'],
    y=knowledgement_drill_ddpgs['mean'],
    mode='lines',
    name='Mean Account Value'
)

trace_upper = go.Scatter(
    x=knowledgement_drill_ddpgs['date'],
    y=knowledgement_drill_ddpgs['mean'] + knowledgement_drill_ddpgs['std'],
    fill=None,
    mode='lines',
    line=dict(color='lightgrey'),
    showlegend=False,
    name='miu + std'
)

trace_lower = go.Scatter(
    x=knowledgement_drill_ddpgs['date'],
    y=knowledgement_drill_ddpgs['mean'] - knowledgement_drill_ddpgs['std'],
    fill='tonexty',  # 填充至上一个跟踪器（即 trace_upper）
    mode='lines',
    line=dict(color='lightgrey'),
    showlegend=False,
    name='miu - std'
)

# 将跟踪器放入数据列表中
data = [trace_mean, trace_upper, trace_lower]

# 设置布局
layout = go.Layout(
    title='Mean Account Value Over Time with Standard Deviation',
    xaxis=dict(title='Date'),
    yaxis=dict(title='Account Value'),
    hovermode='x'
)

# 创建图表
fig = go.Figure(data=data, layout=layout)

# 显示图表
# pyo.plot(fig, filename='knowledgement_drill_ddpg_mean_account_value_with_std.html')  # 保存为HTML文件

def annualized_return(df):
    df['Date'] = pd.to_datetime(df['Date'])  # 转换为 datetime 类型
    total_days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days
    total_years = total_days / 365.25
    return ((df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) ** (1 / total_years) - 1) * 100
def total_return(df):
    return (df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0] - 1) * 100

def annualized_return(df):
    total_years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25
    return ((df['Portfolio_Value'].iloc[-1] / df['Portfolio_Value'].iloc[0]) ** (1 / total_years) - 1) * 100

def sharpe_ratio(df, risk_free_rate=0.02):
    daily_return = df['Portfolio_Value'].pct_change()
    return (daily_return.mean() - risk_free_rate / 252) / daily_return.std() * np.sqrt(252)

def max_drawdown(df):
    roll_max = df['Portfolio_Value'].cummax()
    drawdown = df['Portfolio_Value'] / roll_max - 1.0
    return drawdown.min() * 100

def sortino_ratio(df, risk_free_rate=0.0):
    df['Daily Return'] = df['Portfolio_Value'].pct_change()
    df['Downside Return'] = 0
    df.loc[df['Daily Return'] < 0, 'Downside Return'] = df['Daily Return']**2
    expected_return = df['Daily Return'].mean()
    downside_risk = np.sqrt(df['Downside Return'].mean())
    sortino_ratio = (expected_return - risk_free_rate) / downside_risk
    return sortino_ratio

def beta_value(strategy_df, benchmark_df):
    # 确保日期对齐
    strategy_df = strategy_df.set_index('Date')
    benchmark_df = benchmark_df.set_index('Date')
    common_dates = strategy_df.index.intersection(benchmark_df.index)
    strategy_df = strategy_df.loc[common_dates]
    benchmark_df = benchmark_df.loc[common_dates]

    # 计算日收益率
    strategy_return = strategy_df['Portfolio_Value'].pct_change()
    benchmark_return = benchmark_df['Portfolio_Value'].pct_change()

    # 计算协方差
    covariance = np.cov(strategy_return[1:], benchmark_return[1:])[0][1]
    variance = np.var(benchmark_return[1:])
    beta = covariance / variance
    return beta


def alpha_value(strategy_df, benchmark_df, risk_free_rate=0.0):
    strategy_annual_return = annualized_return(strategy_df)
    benchmark_annual_return = annualized_return(benchmark_df)
    strategy_beta = beta_value(strategy_df, benchmark_df)
    alpha = strategy_annual_return - risk_free_rate - strategy_beta * (benchmark_annual_return - risk_free_rate)
    return alpha

def information_ratio(strategy_df, benchmark_df):
    strategy_return = strategy_df['Portfolio_Value'].pct_change()
    benchmark_return = benchmark_df['Portfolio_Value'].pct_change()
    excess_return = strategy_return - benchmark_return
    tracking_error = excess_return.std()
    information_ratio = excess_return.mean() / tracking_error
    return information_ratio

def calmar_ratio(df):
    annualized_return_value = annualized_return(df)
    max_drawdown_value = max_drawdown(df)
    calmar_ratio = annualized_return_value / abs(max_drawdown_value)
    return calmar_ratio

def win_rate(df):
    df['Daily Return'] = df['Portfolio_Value'].pct_change()
    wins = df[df['Daily Return'] > 0].shape[0]
    total_trades = df['Daily Return'].dropna().shape[0]
    win_rate = wins / total_trades
    return win_rate

def profit_loss_ratio(df):
    df['Daily Return'] = df['Portfolio_Value'].pct_change()
    average_profit = df[df['Daily Return'] > 0]['Daily Return'].mean()
    average_loss = abs(df[df['Daily Return'] < 0]['Daily Return'].mean())
    if average_loss == 0:
        return np.inf
    return average_profit / average_loss

def volatility(df):
    return df['Portfolio_Value'].pct_change().std()

def var_cvar(df, confidence_level=0.05):
    returns = df['Portfolio_Value'].pct_change().dropna()
    var = np.percentile(returns, 100 * confidence_level)
    cvar = returns[returns <= var].mean()
    return var, cvar

# 打印每个条形图的信息
def print_bar_info(bar_container):
    for rect in bar_container:
        height = rect.get_height()
        x = rect.get_x()
        y = rect.get_y()
        width = rect.get_width()
        print(f"Height: {height}, X: {x}, Y: {y}, Width: {width}")

def create_bar_graph(results):
    strategies = results['Strategy']
    total_returns = results['Total Return']
    annualized_returns = results['Annualized Return']

    x = np.arange(len(strategies))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, total_returns, width, label='Total Return')
    rects2 = ax.bar(x + width/2, annualized_returns, width, label='Annualized Return')

    print("Rects1 Info:")
    print_bar_info(rects1)

    print("\nRects2 Info:")
    print_bar_info(rects2)
    # Add some text for labels, title, and custom x-axis tick labels
    ax.set_ylabel('Returns (%)')
    ax.set_title('Returns by Strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.legend()
    plt.savefig("bar.png")
    plt.show()

dji = pd.read_csv("dji.csv")
Markvoize = pd.read_csv("Markowitz.csv")
ddpg = pd.read_csv("originate_ddpg.csv", index_col=0)
aco_optimized_portfolio = pd.read_csv("aco_Optimized_Portfolio.csv")
annealing_Optimized_Portfolio = pd.read_csv("annealing_Optimized_Portfolio.csv")
Anticor_Model = pd.read_csv("Anticor_Model.csv")
BAH = pd.read_csv("BAH_capital_curve.csv")
BCRP = pd.read_csv("BCRP_Model.csv")
Best_k_Model = pd.read_csv("Best_k_Model.csv")
Best_NN_Model = pd.read_csv("Best_NN_Model.csv")
CORN_Model = pd.read_csv("CORN_Model.csv")
CRP_portfolio_values = pd.read_csv("CRP_portfolio_values.csv")
CWMR_Model = pd.read_csv("CWMR_Model.csv")
EG_Model = pd.read_csv("EG_Model.csv")
Genetic_Optimized_Portfolio = pd.read_csv("Genetic_Optimized_Portfolio.csv")
M0_Model = pd.read_csv("M0_Model.csv")
Markowitz_knowledge_drill_capital_curve = pd.read_csv("Markowitz_knowledge_drill_capital_curve.csv")
MVO = pd.read_csv("MVO.csv")
OLMAR_Model = pd.read_csv("OLMAR_Model.csv")
ONS = pd.read_csv("ONS.csv")
PAMR_Model = pd.read_csv("PAMR_Model.csv")
PSO_Optimized_Portfolio = pd.read_csv("PSO_Optimized_Portfolio.csv")
SP_Model = pd.read_csv("SP_Model.csv")
T0_Model = pd.read_csv("T0_Model.csv")
UP = pd.read_csv("UP.csv")
knowledgement_drill_ddpg = knowledgement_drill_ddpgs[['date','mean']]


knowledgement_drill_ddpg.columns = ['Date', 'Portfolio_Value']
MVO.columns = ['Date', 'Portfolio_Value']
dji.columns = ['Date', 'Portfolio_Value']
Markvoize.columns = ['Date', 'Portfolio_Value']
ddpg.columns = ['Date', 'Portfolio_Value']
BAH.columns = ['Date', 'Portfolio_Value']
Markowitz_knowledge_drill_capital_curve.columns = ['Date', 'Portfolio_Value']

# 将所有数据合并到一个DataFrame
strategies = {
    'DJI': dji,
    'Markowitz': Markvoize,
    'DDPG': ddpg,
    'ACO': aco_optimized_portfolio,
    'Annealing': annealing_Optimized_Portfolio,
    'Anticor': Anticor_Model,
    'BAH': BAH,
    'BCRP': BCRP,
    'B^k': Best_k_Model,
    'B^NN': Best_NN_Model,
    'CORN': CORN_Model,
    'CRP': CRP_portfolio_values,
    'CWMR': CWMR_Model,
    'EG': EG_Model,
    'Genetic': Genetic_Optimized_Portfolio,
    'M0': M0_Model,
    'MKD': Markowitz_knowledge_drill_capital_curve,
    'MVO': MVO,
    'OLMAR': OLMAR_Model,
    'ONS': ONS,
    'PAMR': PAMR_Model,
    'PSO': PSO_Optimized_Portfolio,
    'SP': SP_Model,
    'T0': T0_Model,
    'UP': UP,
    'KDD':knowledgement_drill_ddpg
}

# 准备计算结果的DataFrame
results = pd.DataFrame(columns=[
    "Strategy",
    "Total Return",
    "Annualized Return",
    "Sharpe Ratio",
    "Max Drawdown",
    "Sortino Ratio",
    "Beta",
    "Alpha",
    "Information Ratio",
    "Calmar Ratio",
    "Win Rate",
    "Profit/Loss Ratio",
    "Volatility",
    "VaR",
    "CVaR"
])

# benchmark_df = strategies['DJI']  # 设置基准为DJI

for strategy_name, strategy_df in strategies.items():
    strategy_df['Date'] = pd.to_datetime(strategy_df['Date'])  # 转换为 datetime 类型
    benchmark_df = strategies['DJI']  # 作为基准
    benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])

    # 确保数据对齐
    strategy_df = strategy_df.set_index('Date')
    benchmark_df = benchmark_df.set_index('Date')
    common_dates = strategy_df.index.intersection(benchmark_df.index)
    strategy_df = strategy_df.loc[common_dates]
    benchmark_df = benchmark_df.loc[common_dates]

    # 重置索引以便后续计算
    strategy_df.reset_index(inplace=True)
    benchmark_df.reset_index(inplace=True)

    # 计算各项指标
    total_return_value = total_return(strategy_df)
    annualized_return_value = annualized_return(strategy_df)
    sharpe_ratio_value = sharpe_ratio(strategy_df)
    max_drawdown_value = max_drawdown(strategy_df)
    sortino_ratio_value = sortino_ratio(strategy_df)
    beta_val = beta_value(strategy_df, benchmark_df)
    alpha_val = alpha_value(strategy_df, benchmark_df)
    information_ratio_val = information_ratio(strategy_df, benchmark_df)
    calmar_ratio_val = calmar_ratio(strategy_df)
    win_rate_val = win_rate(strategy_df)
    p_l_ratio = profit_loss_ratio(strategy_df)
    vol = volatility(strategy_df)
    var, cvar = var_cvar(strategy_df)

    # 构建一个临时DataFrame来存储结果
    temp_df = pd.DataFrame({
        "Strategy": [strategy_name],
        "Total Return": [total_return_value],
        "Annualized Return": [annualized_return_value],
        "Sharpe Ratio": [sharpe_ratio_value],
        "Max Drawdown": [max_drawdown_value],
        "Sortino Ratio": [sortino_ratio_value],
        "Beta": [beta_val],
        "Alpha": [alpha_val],
        "Information Ratio": [information_ratio_val],
        "Calmar Ratio": [calmar_ratio_val],
        "Win Rate": [win_rate_val],
        "Profit/Loss Ratio": [p_l_ratio],
        "Volatility": [vol],
        "VaR": [var],
        "CVaR": [cvar]
    })

    # 使用 pd.concat 而不是 .append
    results = pd.concat([results, temp_df], ignore_index=True)

create_bar_graph(results)
# Heatmap for performance metrics
plt.figure(figsize=(10, 8))
sns.heatmap(results.set_index('Strategy').drop(['Total Return', 'Annualized Return'], axis=1),
            annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Investment Strategy Performance Metrics')
plt.savefig("Heatmap.png")
plt.show()

# Scatter plot for Risk vs. Return
plt.figure(figsize=(10, 8))
plt.scatter(results['Volatility'], results['Annualized Return'], c=results['Sharpe Ratio'], cmap='RdYlGn')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility (Risk)')
plt.ylabel('Annualized Return')
plt.title('Risk vs Return of Investment Strategies')
for i, txt in enumerate(results.Strategy):
    plt.annotate(txt, (results['Volatility'][i], results['Annualized Return'][i]), fontsize=8)
    print(txt, (results['Volatility'][i], results['Annualized Return'][i]))
plt.savefig("Risk vs Return.png", format='png')
plt.show()

# Select metrics for radar chart
selected_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown', 'Win Rate']
radar_df = results[['Strategy'] + selected_metrics].set_index('Strategy')

# Normalize data for radar chart
radar_df_normalized = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min())

# Number of variables
categories = list(radar_df_normalized)
N = len(categories)

# Radar chart plot
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
plt.ylim(0, 1)

# Plot each strategy
for index, row in radar_df_normalized.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=index)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
plt.savefig("radar chart.png")
plt.show()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(dji['Date'], dji['Portfolio_Value'], label='DJI')
plt.plot(Markvoize['Date'], Markvoize['Portfolio_Value'], label='Markowitz')
plt.plot(ddpg['Date'], ddpg['Portfolio_Value'], label='DDPG')
plt.plot(Markowitz_knowledge_drill_capital_curve['Date'], Markowitz_knowledge_drill_capital_curve['Portfolio_Value'], label='MKD')
plt.plot(knowledgement_drill_ddpg['Date'], knowledgement_drill_ddpg['Portfolio_Value'], label='KDD')

plt.title('Portfolio Value Comparison Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.legend()
plt.savefig("comparison_plot.pdf", format='pdf')
plt.show()


results.to_csv("results.csv")



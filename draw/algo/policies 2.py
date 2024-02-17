from typing import Any, Dict, List, Optional, Type, Union

# new added
from torchsummary import summary
import pandas as pd
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import CovarianceShrinkage
from pypfopt import expected_returns
from finrl import config_tickers
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
# from finrl import config_tickers
from finrl.config import INDICATORS
import itertools
from stable_baselines3.common.logger import configure
# from finrl.agents.stablebaselines3.models import DRLAgent
# from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
# from finrl.main import check_and_make_directories
# from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
# added finish

import torch as th
from gymnasium import spaces
from torch import nn
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

class HuFeatureExtractor(nn.Module):
    def __init__(self, input_features, output_features):
        super(HuFeatureExtractor, self).__init__()
        # 定义网络结构
        self.fc1 = nn.Linear(input_features, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, output_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 最后一层不加激活函数以保留线性特征
        return x

class Actor(BasePolicy):
    """
    Actor network (policy) for TD3.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        actor_net = create_mlp(features_dim, action_dim, net_arch, activation_fn, squash_output=True)
        # Deterministic action

        self.mu = nn.Sequential(*actor_net)


        # processed_full = pd.read_csv("/Users/ganghu/Desktop/pythonProject1/raw_data.csv", parse_dates=['date'])
        #
        # # 定义要保留的股票代码列表
        # tickers_to_keep = ['AAPL', 'CRM', 'V', 'UNH', 'BA']#
        #
        # # 筛选出符合条件的行
        # processed_full = processed_full[processed_full['tic'].isin(tickers_to_keep)]
        #
        # processed_full = processed_full.drop(columns=['turbulence'])
        #
        # # 要删除的列名列表
        # columns_to_delete = ['open', 'high', 'low', 'volume', 'day', 'macd',
        #                      'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma',
        #                      'close_60_sma', 'vix']
        #
        # # 删除这些列
        # processed_full.drop(columns=columns_to_delete, inplace=True)
        #
        # # 获取唯一的 tic 值
        # tics = processed_full['tic'].unique()
        #
        # # 为每个 tic 创建一个新列
        # for tic in tics:
        #     processed_full[f'weight_{tic}'] = np.where(processed_full['tic'] == tic, processed_full['weight'], 0)
        #     processed_full[f'close_{tic}'] = np.where(processed_full['tic'] == tic, processed_full['close'], 0.0)
        #
        # # 可选：删除原始的 weight 列
        # processed_full.drop(columns=['weight'], inplace=True)
        #
        # # 分割数据集
        # train_df = processed_full[(processed_full['date'] >= '2009-01-01') & (processed_full['date'] < '2015-01-01')]
        # val_df = processed_full[(processed_full['date'] >= '2015-01-01') & (processed_full['date'] < '2016-01-01')]
        # test_df = processed_full[processed_full['date'] >= '2016-01-01']
        #
        # # 将 Pandas DataFrame 转换为 PyTorch 张量
        # def dataframe_to_tensor(df):
        #     # 处理日期数据
        #     df['year'] = df['date'].dt.year
        #     df['month'] = df['date'].dt.month
        #     df['day'] = df['date'].dt.day
        #     df['weekday'] = df['date'].dt.weekday
        #
        #     # 获取所有唯一的 tic 值
        #     tics = df['tic'].unique()
        #
        #     # 生成 close_{tic} 列的名称
        #     close_columns = [f'close_{tic}' for tic in tics]
        #
        #     # 处理股票代码：one-hot encoding
        #     tic_dummies = pd.get_dummies(df['tic'], prefix='tic').astype(float)
        #     df = pd.concat([df, tic_dummies], axis=1)
        #
        #     df = df.select_dtypes(include=[np.number])
        #
        #     # 选择输入列（包括新的日期和股票代码特征）
        #     X_columns = ['year', 'month', 'day', 'weekday'] + \
        #                 [f'{tic}' for tic in tic_dummies.columns] + \
        #                 close_columns  # 添加所有需要的列
        #     X = df[X_columns].values
        #
        #     # 获取所有以 'weight_' 开头的列名
        #     weight_columns = [col for col in df.columns if col.startswith('weight_')]
        #
        #     # 选择输出列
        #     # 选取这些列作为 y
        #     y = df[weight_columns].values
        #
        #     return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32),len(X_columns)
        #
        # X_train, y_train,length = dataframe_to_tensor(train_df)
        # X_val, y_val,_ = dataframe_to_tensor(val_df)
        # X_test, y_test,_ = dataframe_to_tensor(test_df)
        #
        # scaler = StandardScaler()
        # X_train_scaled = scaler.fit_transform(X_train)
        # X_val_scaled = scaler.transform(X_val)
        # X_test_scaled = scaler.transform(X_test)
        #
        # # 将规范化后的数据转换为 PyTorch 张量
        # X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        # X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        # X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        #
        # # ******************************定义网络************************************
        # feature_extractor_hu = HuFeatureExtractor(length,51)# 291
        # model_fresh = nn.Sequential(*actor_net)
        #
        # # ******************************打印网络形式************************************
        # summary(model_fresh, (features_dim,))
        #
        # for name, param in model_fresh.named_parameters():
        #     print(f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}")
        #
        # # ******************************打印完毕************************************
        #
        # batch_size = 128  # 或您选择的其他批大小
        #
        # # train_dataset = TensorDataset(X_train_scaled, y_train)
        # # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # #
        # # val_dataset = TensorDataset(X_val_scaled, y_val)
        # # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        #
        # # 创建 DataLoader
        # train_dataset = TensorDataset(X_train_scaled_tensor, y_train)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        #
        # val_dataset = TensorDataset(X_val_scaled_tensor, y_val)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        #
        # criterion = nn.MSELoss()  # 假设回归任务使用均方误差
        # optimizer = optim.Adam(model_fresh.parameters(), lr=0.00001)  # 可以调整学习率
        #
        # epochs = 50  # 或您选择的其他周期数
        #
        # for epoch in range(epochs):
        #     model_fresh.train()
        #     for X_batch, y_batch in train_loader:
        #         # print("X_batch stats: Mean =", X_batch.mean().item(), "Std =", X_batch.std().item())
        #         # print("Original X_batch shape:", X_batch.shape)
        #         X_batch_transformed = feature_extractor_hu(X_batch)  # 转换后的数据
        #         # print("Transformed X_batch shape:", X_batch_transformed.shape)
        #
        #         # 前向传播
        #         y_pred = model_fresh(X_batch_transformed)
        #         # print("y_pred stats: Mean =", y_pred.mean().item(), "Std =", y_pred.std().item())
        #         # print("Model output y_pred shape:", y_pred.shape)
        #         loss = criterion(y_pred, y_batch)
        #         # print("Loss:", loss.item())
        #
        #         # 反向传播和优化
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #     # 验证集评估
        #     model_fresh.eval()
        #     val_loss_sum = 0
        #     with torch.no_grad():
        #         for X_val_batch, y_val_batch in val_loader:
        #             # 应用特征提取器
        #             X_val_batch_transformed = feature_extractor_hu(X_val_batch)
        #             # 预测
        #             y_val_pred = model_fresh(X_val_batch_transformed)
        #             # 计算损失
        #             val_loss = criterion(y_val_pred, y_val_batch)
        #             val_loss_sum += val_loss.item()
        #
        #     average_val_loss = val_loss_sum / len(val_loader)
        #
        #     print(f"Epoch {epoch}, Loss: {loss.item()}, Average Val Loss: {average_val_loss}")
        #     # print("Current learning rate:", optimizer.param_groups[0]['lr'])
        #
        # # print("Model weights:")
        # # for name, param in model_fresh.named_parameters():
        # #     if param.requires_grad:
        # #         print(name, param.data)
        #
        # self.mu = model_fresh

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs, self.features_extractor)
        return self.mu(features)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self(observation)


class TD3Policy(BasePolicy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
            normalize_images=normalize_images,
        )

        # Default network architecture, from the original paper
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = [256, 256]
            else:
                net_arch = [400, 300]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        # Create actor and target
        # the features extractor should not be shared
        self.actor = self.make_actor(features_extractor=None)
        self.actor_target = self.make_actor(features_extractor=None)
        # Initialize the target to have the same weights as the actor
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Critic target should not share the features extractor with critic
            # but it can share it with the actor target as actor and critic are sharing
            # the same features_extractor too
            # NOTE: as a result the effective poliak (soft-copy) coefficient for the features extractor
            # will be 2 * tau instead of tau (updated one time with the actor, a second time with the critic)
            self.critic_target = self.make_critic(features_extractor=self.actor_target.features_extractor)
        else:
            # Create new features extractor for each network
            self.critic = self.make_critic(features_extractor=None)
            self.critic_target = self.make_critic(features_extractor=None)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = self.optimizer_class(
            self.critic.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.actor_target.set_training_mode(False)
        self.critic_target.set_training_mode(False)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                share_features_extractor=self.share_features_extractor,
            )
        )
        return data

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._predict(observation, deterministic=deterministic)

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # Note: the deterministic deterministic parameter is ignored in the case of TD3.
        #   Predictions are always deterministic.
        return self.actor(observation)

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode


MlpPolicy = TD3Policy


class CnnPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )


class MultiInputPolicy(TD3Policy):
    """
    Policy class (with both actor and critic) for TD3 to be used with Dict observation spaces.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = CombinedExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor,
        )

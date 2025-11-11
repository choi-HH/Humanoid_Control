import os
import copy

import torch
import torch.nn as nn
from torch.distributions import Normal # 확률분포 관련 클래스 (정규분포 Normal Distribution) 
from .utils import create_MLP, weights_init_
from .utils import RunningMeanStd

# Actor network
class Actor(nn.Module):
    def __init__(self,
                 num_obs, # actor obs 차원 (input)
                 num_actions, # action 차원 (output)
                 hidden_dims,
                 activation="elu",
                 init_noise_std=1.0, # 초기 노이즈 표준편차
                 normalize_obs=False,
                 log_std_bounds=None,
                 actions_limits=None,
                 custom_initialization=False,
                 **kwargs):

        if kwargs:
            print("Actor.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs) # RunningMeanStd: 관측치 정규화에 사용되는 클래스

        self.mean_NN = create_MLP(num_obs, num_actions, hidden_dims, activation) # 평균(μ) 신경망 구축
        self.log_std_NN = None

        # Action noise
        if log_std_bounds is not None:
            self.log_std_min, self.log_std_max = log_std_bounds
            self.log_std_NN = create_MLP(num_obs, num_actions, hidden_dims, activation)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        self.distribution = None

        if actions_limits is not None:
            self.actions_min, self.actions_max = actions_limits
            self.actions_range_center = (self.actions_max + self.actions_min) / 2
            self.actions_range_radius = (self.actions_max - self.actions_min) / 2

        # disable args validation for speedup
        Normal.set_default_validate_args = False
        if custom_initialization:
            self.apply(weights_init_)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # 정규 분포 업데이트
    def update_distribution(self, observations):
        if self._normalize_obs: # 관측치 정규화
            observations = self.norm_obs(observations) # 정규화된 관측치 반환
        # ===== 평균 계산 =====
        # 관측값을 평균 신경망에 통과시켜 가장 좋은 행동(평균) 계산
        mean = self.mean_NN(observations)
        # ====================

        # ===== 분산(표준편차) 계산 =====
        # 분산 신경망이 없으면 고정된 표준편차 사용
        if self.log_std_NN is None:
            self.distribution = Normal(mean, mean*0. + self.std) # 의미: 평균은 mean, 표준편차는 고정된 self.std 사용
                                                                # self.std는 nn.Parameter로 정의되어 학습 가능
                                                                # nn.Parameter: 텐서를 래핑하여 모델의 학습 가능한 매개변수로 만듦
                                                                # mean*0. : mean과 같은 크기의 텐서를 만들기 위한 연산 (브로드캐스팅)
                                                                # mean*0. + self.std : 모든 배치에 대해 동일한 표준편차 사용

        # 분산 신경망이 있으면 관측값을 분산 신경망에 통과시켜 로그 표준편차 계산
        else: # TODO: Implement s.t. mean & log_std shares parameters only last layer is different!
            log_std = self.log_std_NN(observations)
            log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
            self.std = torch.exp(log_std)
            self.distribution = Normal(mean, mean*0. + self.std)

    # action (output)
    def act(self, observations):
        self.update_distribution(observations) # obs 받아서 action 생성
        return self.distribution.sample() # 샘플링된 행동 반환
    
    def ract(self, observations):
        """ Sample with reparametrization trick """
        self.update_distribution(observations)
        return self.distribution.rsample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_scaled_ractions_and_log_prob(self, observations, only_actions=False):
        """ Get scaled actions using reparametrization trick and their log probability
            Implemented solely for SAC """ 
        self.update_distribution(observations)
        actions = self.distribution.rsample()
        actions_normalized = torch.tanh(actions)
        actions_scaled = (self.actions_range_center + self.actions_range_radius * actions_normalized)
        
        if only_actions:
            return actions_scaled 
        else:
            actions_log_prob = self.distribution.log_prob(actions).sum(dim=-1) - \
                               torch.log(1.0 - actions_normalized.pow(2) + 1e-6).sum(-1)
            return actions_scaled, actions_log_prob

    def act_inference(self, observations):
        if self._normalize_obs:
            observations = self.norm_obs(observations)
        actions_mean = self.mean_NN(observations)
        return actions_mean

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.obs_rms(observation)
        
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path_TS = os.path.join(path, 'policy.pt') # TorchScript path
        path_onnx = os.path.join(path, 'policy.onnx') # ONNX path

        if self._normalize_obs:
            class NormalizedActor(nn.Module):
                def __init__(self, actor, obs_rms):
                    super().__init__()
                    self.actor = actor
                    self.obs_rms = obs_rms
                def forward(self, obs):
                    obs = self.obs_rms(obs)
                    return self.actor(obs)
            model = NormalizedActor(copy.deepcopy(self.mean_NN), copy.deepcopy(self.obs_rms)).to('cpu')
        
        else:
            model = copy.deepcopy(self.mean_NN).to('cpu')

        dummy_input = torch.rand(self.mean_NN[0].in_features,)
        model_traced = torch.jit.trace(model, dummy_input)
        torch.jit.save(model_traced, path_TS)
        torch.onnx.export(model_traced, dummy_input, path_onnx)

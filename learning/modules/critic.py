import torch
import torch.nn as nn
from .utils import create_MLP, weights_init_
from .utils import RunningMeanStd

# Critic network
class Critic(nn.Module):
    def __init__(self,
                 num_obs, # critic obs 차원 (input)
                 hidden_dims,
                 activation="elu",
                 normalize_obs=False,
                 custom_initialization=False,
                 **kwargs):

        if kwargs:
            print("Critic.__init__ got unexpected arguments, "
                  "which will be ignored: "
                  + str([key for key in kwargs.keys()]))
        super().__init__()

        self.NN = create_MLP(num_obs, 1, hidden_dims, activation) # Critic 신경망 구축

        # 관측치 정규화
        self._normalize_obs = normalize_obs
        if self._normalize_obs:
            self.obs_rms = RunningMeanStd(num_obs)

        if custom_initialization:
            self.apply(weights_init_)

    # value 함수 평가 (output)
    def evaluate(self, critic_observations, actions=None):
        if actions is None: # 액션이 주어지지 않으면
            if self._normalize_obs: # 관측치 정규화
                critic_observations = self.norm_obs(critic_observations) # 정규화된 관측치 반환
            """
                학습에서 이 상황이 얼마나 좋은가? 를 평가하는 value function 
                1. V(s)-function: 상태 s 만으로 얼마나 좋은가 평가
                2. Q(s,a)-function: 상태 s 와 액션 a 로 얼마나 좋은가 평가
            """
            # Critic is V(s)-function estimator
            return self.NN(critic_observations)
        else:
            # Critic is Q(s,a)-function estimator
            concat_input = torch.cat((critic_observations, actions), dim=1)
            return self.NN(concat_input)

    def norm_obs(self, observation):
        with torch.no_grad():
            return self.obs_rms(observation)

    def freeze_parameters(self):
        for parameters in self.NN.parameters():
            parameters.requires_grad = False

    def update_parameters(self, src_model: 'Critic', polyak: float):
        with torch.inference_mode():
            for parameters, src_parameters in zip(self.NN.parameters(), src_model.NN.parameters()):
                parameters.data.mul_(1 - polyak)
                parameters.data.add_(polyak * src_parameters.data)

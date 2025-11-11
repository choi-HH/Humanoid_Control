import torch
import torch.nn as nn
from .utils import create_MLP

class Estimator(nn.Module):
    def __init__(self,
                 num_obs_history, # Estimator가 과거 몇 개의 관측치를 볼지
                 num_non_privileged_obs, # 측정가능한 obs (joint 각도, 속도, base orientation 등)
                 num_privileged_obs, # 측정불가능한 obs의 차원 (base_height, base_vel 등)
                 latent_dim, # Estimator가 출력하는 latent vector의 차원
                 hidden_dims=[256, 256],
                 activation="elu",
                 **kwargs):

        super(Estimator, self).__init__()

        self.num_obs_history = num_obs_history
        self.num_non_privileged_obs = num_non_privileged_obs

        # 입력 차원: (과거 관측치 수10개를 본다 * 비특권 관측치 수 66개의 관측치)
        input_dim = num_obs_history * num_non_privileged_obs

        # 1. MLP (해당 Estimator는 몸통(base_NN)은 하나고 머리(head)는 2개임) 
        # non_privileged_obs_history를 받아서 latent representation으로 매핑
        self.base_NN = create_MLP(input_dim, hidden_dims[-1], hidden_dims[:-1], activation)

        # 2. latent vector output header
        # 해당 latent vector가 Actor의 입력으로 사용될 수 있도록 변환
        self.latent_head = nn.Linear(hidden_dims[-1], latent_dim)

        # 3. privileged observation 예측 헤더
        # 이 예측값은 Estimator의 loss을 계산하는 데 사용됨
        self.prediction_head = nn.Linear(hidden_dims[-1], num_privileged_obs)

    def forward(self, obs_history):
        """
        obs history -> base NN(estimator network) -> latent vector + predicted states

        Args:
            obs_history (torch.Tensor): [batch_size, num_obs_history * num_non_privileged_obs]
                                        (여기선 flatten된 입력을 가정)

        Returns:
            torch.Tensor: latent_vector [batch_size, latent_dim]
            torch.Tensor: predicted_states [batch_size, num_privileged_obs]
        """
        
        # 입력이 (batch_size, history, obs_dim) 형태인 경우를 대비해 flatten
        if obs_history.dim() > 2:
            # 입력이 [batch_size, 10, 66] 형태인 경우 [batch_size, 660] 형태로 변환
            obs_history = torch.flatten(obs_history, start_dim=1)
        # MLP 통과
        base_output = self.base_NN(obs_history)
        # latent vector 생성
        latent_vector = self.latent_head(base_output)
        predicted_states = self.prediction_head(base_output)

        return latent_vector, predicted_states
    
    def get_latent(self, obs_history):
        """
        depoly 시에는 latent_vector만 필요 
        """
        if obs_history.dim() > 2:
            obs_history = torch.flatten(obs_history, start_dim=1)

        base_output = self.base_NN(obs_history)
        latent_vector = self.latent_head(base_output)
        return latent_vector
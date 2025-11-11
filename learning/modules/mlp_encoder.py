"""
import torch
import torch.nn as nn # 신경망 모듈, 레이어, 활성화 함수 등 가져옴

# Estimator network
class MLP_Encoder(nn.Module):
    def __init__(self,
                 num_input_dim,  # obs history 차원 (예: 660)
                 num_output_dim, # latent vector 차원 (예: 32)
                 hidden_dims=[256, 256],
                 activation="elu",
                 **kwargs): # 추가 인자는 받지 않음

        super(MLP_Encoder, self).__init__()

        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim # 클래스 내부 변수

        activation = self.get_activation(activation) # 활성화 함수

        # MLP 구축
        # 신경망 레이어들을 담을 빈 리스트
        encoder_layers = []
        
        # 1. 입력층 (Input -> Hidden1)
        encoder_layers.append(nn.Linear(self.num_input_dim, hidden_dims[0])) # 첫 번째 은닉층으로 가는 선형 변환 레이어 추가
        encoder_layers.append(activation) # 활성화 함수 추가

        # 의미: 입력 데이터를 서로 다른 가중치로 조합하여 중간 데이터를 만들고, 활성화 함수를 적용하여 비선형성을 부여(선택적 정보 전달)함.

        # 2. 중간 은닉층들 (Hidden1 -> Hidden2 ... -> LastHidden)
        for l in range(len(hidden_dims) - 1): # 마지막 은닉층에서 출력층으로 가기 전까지
            encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1])) # 선형 변환 레이어 추가
            encoder_layers.append(activation)

        # 의미: 여러 은닉층을 거치면서 점점 더 추상적이고 복잡한 특징들을 학습함.


        # 3. 출력층 (LastHidden -> Output)
        encoder_layers.append(nn.Linear(hidden_dims[-1], self.num_output_dim))
        
        # 리스트에 담은 모든 레이어를 nn.Sequential로 합쳐서 하나의 'encoder' 모듈로 만듦
        self.encoder = nn.Sequential(*encoder_layers)

        print(f"Encoder MLP: {self.encoder}")
        # ==========================================================

    def forward(self, input_tensor):

        ### 입력이 이미 [batch_size, num_input_dim] 형태로 쭉 펴져있다고 가정함.###
        # Args:
        #     input_tensor (torch.Tensor): [batch_size, num_input_dim] (예: [b, 660])

        # Returns:
        #     torch.Tensor: latent_vector [batch_size, num_output_dim] (예: [b, 32])

        return self.encoder(input_tensor)

    # get_activation 헬퍼 함수
    def get_activation(self, act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            print("invalid activation function!")
            return None
"""
# 위는 tron1 스타일 코딩
import torch
import torch.nn as nn # 신경망 모듈, 레이어, 활성화 함수 등 가져옴
from .utils import create_MLP

# Estimator network
class MLP_Encoder(nn.Module):
    def __init__(self,
                 num_input_dim,  # obs history 차원
                 num_output_dim, # latent vector 차원
                 hidden_dims=[256, 256],
                 activation="elu",
                 **kwargs): # 추가 인자는 받지 않음
        super().__init__()

        self.num_input_dim = num_input_dim
        self.num_output_dim = num_output_dim

        # MLP 구축
        self.encoder = create_MLP(num_input_dim, num_output_dim, hidden_dims, activation)

        print(f"Encoder MLP: {self.encoder}")
        # ==========================================================

    # estimator 네트워크의 순전파
    def forward(self, input_tensor):
        return self.encoder(input_tensor)

    # estimator network 호출
    def encode(self, input):
        self.encoder_out = self.encoder(input)
        return self.encoder_out
    """
        self.encoder.encode(obs_history_batch) = self.encoder(obs_history_batch)
        위와 같이 작성하면 forward와 encode 함수 역할이 같음.
        다만, encode라는걸 명시하기 위해 별도의 함수를 만든 것임.
    """
    
    # encoder output 반환
    def get_encoder_out(self):
        return self.encoder_out
    
    # 
    def inference(self, input):
        with torch.no_grad(): # gradient 계산 비활성화
            return self.encoder(input)
    """
        기본적인 학습의 흐름.
        처음 환경과 상호작용하지 않고(학습시작) gradient를 업데이트하면서 batch 단위로 데이터를 얻음. <- 이때 mini-batch 단위로 gradient 계산이 활성화됨.
        <mini-batch 개수가 4라면 4번의 forward, backward 연산이 발생함>
        이후 
        환경과 상호작용할 때(학습결과를 로봇에 적용)는 torch.no_grad()가 되어 gradient 계산이 비활성화됨. <- backward 연산이 발생하지 않음.
        따라서, inference 함수는 환경과 상호작용할 때 사용됨.
    """
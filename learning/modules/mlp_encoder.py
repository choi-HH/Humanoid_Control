# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin


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
        """
        의미: 입력 데이터를 서로 다른 가중치로 조합하여 중간 데이터를 만들고, 활성화 함수를 적용하여 비선형성을 부여(선택적 정보 전달)함.
        """

        # 2. 중간 은닉층들 (Hidden1 -> Hidden2 ... -> LastHidden)
        for l in range(len(hidden_dims) - 1): # 마지막 은닉층에서 출력층으로 가기 전까지
            encoder_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1])) # 선형 변환 레이어 추가
            encoder_layers.append(activation)
        """
        의미: 여러 은닉층을 거치면서 점점 더 추상적이고 복잡한 특징들을 학습함.
        """

        # 3. 출력층 (LastHidden -> Output)
        encoder_layers.append(nn.Linear(hidden_dims[-1], self.num_output_dim))
        
        # 리스트에 담은 모든 레이어를 nn.Sequential로 합쳐서 하나의 'encoder' 모듈로 만듦
        self.encoder = nn.Sequential(*encoder_layers)

        print(f"Encoder MLP: {self.encoder}")
        # ==========================================================

    def forward(self, input_tensor):
        """
        입력이 이미 [batch_size, num_input_dim] 형태로
        쭉 펴져있다고 가정함.
        Args:
            input_tensor (torch.Tensor): [batch_size, num_input_dim] (예: [b, 660])

        Returns:
            torch.Tensor: latent_vector [batch_size, num_output_dim] (예: [b, 32])
        """
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
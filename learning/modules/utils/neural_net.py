import torch.nn as nn
from torch.distributions import Normal
import torch

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        # torch.nn.init.constant_(m.bias, 0)

# Create a Multi-Layer Perceptron
def create_MLP(num_inputs, num_outputs, hidden_dims, activation,
               dropouts=None):

    activation = get_activation(activation) # 활성화 함수 객체 얻기

    if dropouts is None: # 기본값 설정
        dropouts = [0]*len(hidden_dims) # 드롭아웃 비율 0

    layers = []
    if not hidden_dims:  # handle no hidden layers
        add_layer(layers, num_inputs, num_outputs) # 출력층만 추가
    else:
        add_layer(layers, num_inputs, hidden_dims[0], activation, dropouts[0]) # 입력층 -> 첫 은닉층
        for i in range(len(hidden_dims)): # 은닉층들 순회
            if i == len(hidden_dims) - 1: # 마지막 은닉층인 경우
                add_layer(layers, hidden_dims[i], num_outputs) # 출력층 추가
            else:
                add_layer(layers, hidden_dims[i], hidden_dims[i+1],
                          activation, dropouts[i+1]) # 은닉층들 추가
    return nn.Sequential(*layers) # 레이어들을 하나의 모듈로 묶어서 반환

# 헬퍼 함수: 활성화 함수 객체 반환
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
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

# 헬퍼 함수: 레이어 추가
def add_layer(layer_list, num_inputs, num_outputs, activation=None, dropout=0):
    layer_list.append(nn.Linear(num_inputs, num_outputs))
    if dropout > 0:
        layer_list.append(nn.Dropout(p=dropout))
    if activation is not None:
        layer_list.append(activation)
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
import torch.nn as nn
import torch.optim as optim

from learning.modules import MLP_Encoder
from learning.modules import ActorCritic
from learning.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    encoder: MLP_Encoder

    def __init__(self,
                 encoder,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 est_learning_rate=1.0e-3, # encoder learning rate
                 critic_take_latent=False, # critic가 encoder 출력값을 입력으로 받는지 여부
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.critic_take_latent = critic_take_latent

        self.encoder = encoder

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        # actor-critic optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # encoder optimizer
        if self.encoder.num_output_dim != 0: # encoder가 사용되는 경우
            self.extra_optimizer = optim.Adam(self.encoder.parameters(), lr=est_learning_rate)
        else:
            self.extra_optimizer = None          

        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(
        self, num_envs, 
        num_transitions_per_env, 
        num_obs, # action obs
        num_critic_obs,
        num_obs_history, # + history obs
        num_actions
    ):
        self.storage = RolloutStorage(
            num_envs, 
            num_transitions_per_env, 
            num_obs, # action obs
            num_critic_obs, 
            num_obs_history, # + history obs
            num_actions,  
            self.device
        )

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    # obs 받아서 action 생성
    def act(self, obs, obs_history, critic_obs):
        # Compute the actions and values
        
        # act
        encoder_out = self.encoder(obs_history) # input obs_history --> encoder --> output latent vector
        # self.transition.actions = self.actor_critic.act(obs).detach()
        # encoder 출력값과 obs를 합쳐서 actor-critic 중 actor에 입력
        self.transition.actions = self.actor_critic.act(torch.cat((encoder_out, obs), dim=-1)).detach() # dim=-1: 데이터의 개수는 유지 하면서 마지막 차원 기준으로 합침
        # .detach()를 호출하여 역전파 그래프에서 분리 (gradient 계산 비활성화) --> 데이터만 저장

        # evaluate
        if self.critic_take_latent: # critic가 encoder 출력값을 입력으로 받는 경우
            critic_obs = torch.cat((encoder_out, critic_obs), dim=-1) # 이때 critic_obs에도 encoder 출력값 추가되어 critic obs 차원이 증가되면 
                                                                      # actor_critic가 초기화될 때 critic obs 차원을 맞춰줘야 함
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach() # critic 평가값 저장

        # storage
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = torch.cat((encoder_out, obs), dim=-1) # encoder 출력값 + obs가 actor의 input # obs(이전코드)
        self.transition.critic_observations = critic_obs
        self.transition.observations_history = obs_history # obs_history 추가
        return self.transition.actions

    def process_env_step(self, rewards, dones, timed_out=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if timed_out is not None:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * timed_out.unsqueeze(1), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        # mini-batch generator
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)        
        for obs_batch, critic_obs_batch, obs_history_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                """
                    ex): mini batch size = 128
                    obs_batch: (128, obs_dim), obs_history_batch: (128, obs_history_dim), critic_obs_batch: (128, critic_obs_dim).....
                    과 같이 obs, obs history 데이터 묶음이 통째로 하나의 미니배치로 저장.
                """

                # encoder
                encoder_out_batch = self.encoder(obs_history_batch) # obs_history를 통한 encoder batch 출력값
                self.actor_critic.act(obs_batch) # actor 입력에 encoder 출력값이 합쳐진 obs 합쳐서 입력

                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)

                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # ========== KL ========================================================================================
                if self.desired_kl != None and self.schedule == 'adaptive': # adaptive 스케줄링인 경우(desired 정책변화량(kl)값이 주어진 경우)
                    with torch.inference_mode(): # gradient 계산 비활성화(정책 평가 목적)
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)
                        """
                            옛날 정책의 행동 분포(데이터 수집 시점): N(old_mu_batch, old_sigma_batch^2)
                            현재 정책의 행동 분포(방금 act로 계산한): N(mu_batch, sigma_batch^2)
                            이 두 확률 분포가 얼마나 달라졌는지 측정하는 척도: KL 발산
                            kl_mean = E[KL(N(old_mu_batch, old_sigma_batch^2) || N(mu_batch, sigma_batch^2))]


                            KL(N(old_mu_batch, old_sigma_batch^2) || N(mu_batch, sigma_batch^2))
                            = log(sigma_batch / old_sigma_batch) + (old_sigma_batch^2 + (old_mu_batch - mu_batch)^2) / (2 * sigma_batch^2) - 0.5
                        """

                        if kl_mean > self.desired_kl * 2.0: # 정책 변화량이 너무 큰 경우
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5) # learning rate 1.5배 감소
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0: # 정책 변화량이 너무 작은 경우
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5) # learning rate 1.5배 증가
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate # optimizer의 learning rate 갱신
                # =======================================================================================================

                # ========== Surrogate loss (Actor loss calculation) ====================================================
                """
                    loss function calculation
                    Surrogate인 이유: PPO 논문에서 제안한 clipped objective function이 기존의 policy gradient loss function을 근사(대체)하기 때문.

                    목표: 좋았던 행동(advantages > 0)의 확률을 높이고, 나빴던 행동(advantages < 0)의 확률을 낮춤.
                """
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)) # ratio = π_new(a|s) / π_old(a|s) = exp(log(π_new(a|s)) - log(π_old(a|s)))
                surrogate = -torch.squeeze(advantages_batch) * ratio # loss = -A * ratio
                """
                    ** clipping 안한 경우 **
                    optimization 방향: loss를 최소화하는 방향으로 파라미터 갱신
                    Advantage > 0 -> loss = -(+A) * ratio -> 최소화를 위해 음수 값을 더 작게(더 큰 음수)하기 위해 ratio를 키움 
                        -> ratio가 커지면 loss가 작아짐 -> π_new(a|s) 커짐 -> 좋았던 행동 확률 증가
                    Advantage < 0 -> loss = -(-A) * ratio -> 최소화를 위해 양수 값을 더 작게(더 작은 양수)하기 위해 ratio를 줄임
                        -> ratio가 작아지면 loss가 작아짐 -> π_new(a|s) 작아짐 -> 나빴던 행동 확률 감소
                """
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                """
                    ** clipping 한 경우 **
                    ratio가 너무 커지거나 작아지는 것을 방지하기 위해 클리핑 적용
                    loss = -A * clip(ratio, 1-ε, 1+ε)
                    ε: clip_param
                """
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                """
                    둘 중 더 큰 값을 선택하여 평균을 냄
                    surrogate loss = E[max(-A * ratio, -A * clip(ratio, 1-ε, 1+ε))]
                """
                # ==========================================================================================================

                # ========== Value function loss (Critic loss calculation) ====================================================
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param) # V_new(s) 클리핑
                    value_losses = (value_batch - returns_batch).pow(2) # V_new(s)와 V_target(s)의 MSE
                    value_losses_clipped = (value_clipped - returns_batch).pow(2) # 클리핑된 V_new(s)와 V_target(s)의 MSE
                    value_loss = torch.max(value_losses, value_losses_clipped).mean() # 둘 중 더 큰 값을 선택하여 평균을 냄
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    """
                        value_batch: critic가 예측한 V_new(s)
                        returns_batch: GAE로 계산된 V_target(s)에 해당

                        loss = MSE(V_new(s), V_target(s)) = (V_new(s) - V_target(s))^2
                    """
                # ==========================================================================================================

                # ========== Total loss ===================================================================================
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                """
                    총 loss = surrogate loss + value loss - entropy bonus
                    surrogate loss: 내 Action을 어떻게 바꿔야 Advantage를 더 얻을 수 있을까?
                    value_loss_coef: value loss의 중요도 조절 계수
                    value loss: 내 State value 예측을 얼마나 정확하게 할 수 있을까?
                    entropy bonus: 정책의 무작위성을 촉진하여 탐험 장려
                """
                # ==========================================================================================================

                # ========== Gradient step =================================================================================
                self.optimizer.zero_grad() # gradient 초기화
                loss.backward() # loss에 대한 gradient 계산
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm) # gradient clipping
                self.optimizer.step() # 파라미터 갱신

                mean_value_loss += value_loss.item() # value loss 누적
                mean_surrogate_loss += surrogate_loss.item() # surrogate loss 누적
                # ==========================================================================================================
        
        # encoder optimization step
        num_updates_extra = 0 # encoder 업데이트 횟수 누적
        mean_extra_loss = 0 # encoder loss 누적
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs) # encoder 전용 미니배치 생성기
            for (critic_obs_batch, obs_history_batch) in generator:       
                """
                    critic_obs_batch: (128, critic_obs_dim), obs_history_batch: (128, obs_history_dim)
                    정답지 critic_obs_batch
                """
                # encoder
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out() # estimator의 output값인 lin vel을 encode_batch로 할당
                
                if self.encoder.is_mlp_encoder:
                    extra_loss = ((encode_batch - critic_obs_batch[:, 1:4]).pow(2).mean())
                    # encoder가 예측한 base vel encode_batch와 실제 base vel(critic_obs_batch[:, 1:4]의 MSE
                else:
                    extra_loss = torch.zeros_like(value_loss) # encoder가 없는 경우 loss 0으로 설정
                
                self.extra_optimizer.zero_grad() # gradient 초기화(encoder 전용)
                extra_loss.backward() # encoder loss에 대한 gradient 계산
                self.extra_optimizer.step() # encoder 파라미터 갱신

                num_updates_extra += 1 # encoder 업데이트 횟수 누적
                mean_extra_loss += extra_loss.item() # encoder loss 누적

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra
        mean_surrogate_loss /= num_updates # 이번 에포크의 평균 loss 값 반환
        self.storage.clear()

        return mean_value_loss, mean_extra_loss, mean_surrogate_loss

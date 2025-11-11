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

from .base_config import BaseConfig

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096  # (n_robots in Rudin 2021 paper - batch_size = n_steps * n_robots)  # 4096개의 환경(로봇)개수, GPU에서 병렬로 학습
        num_actuators = 12 # 다리 자유도
        env_spacing = 3.  # not used with heightfields/trimeshes 
        episode_length_s = 20  # episode length in seconds

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh  # 지형 타입
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True   # 지형 커리큘럼 학습 사용 처음엔 쉬운 평지에서 시작해서, 점점 어려운 지형으로 변경
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True # True: 로봇이 지형 스캔 가능. False: 로봇이 지형 스캔 불가(blind walking)
        measured_points_x_range = [-0.6, 0.6] # x range for height measurements
        measured_points_x_num_sample = 13 # 13개 점으로 쪼개서 측정
        measured_points_y_range = [-0.6, 0.6]
        measured_points_y_num_sample = 13 
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        platform_size = 5.
        num_rows= 10 # number of terrain rows (levels)
        num_cols = 10 # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    class commands:
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # default: lin_vel_x, lin_vel_y, yaw_vel => RL 에이전트가 추적해야 하는 명령어의 수
        resampling_time = 10. # time before command are changed[s] 10초마다 명령어 변경
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = 1.   # min max [m/s]
            yaw_vel = 1.    # min max [rad/s]

    class init_state:

        # * target state when actuation = 0, also reset positions for basic mode 
        # *  default_dof_angles 로봇의 관절 초기 각도
        default_joint_angles = {"joint_a": 0.,
                                "joint_b": 0.}

        reset_mode = "reset_to_basic" 
        # reset setup chooses how the initial conditions are chosen. 
        # "reset_to_basic" = a single position 같은 위치와 속도 
        # "reset_to_range" = uniformly random from a range defined below 매번 무작위 위치와 속도

        # * root defaults 로봇 스폰될 때의 초기 위치/속도
        pos = [0.0, 0.0, 1.] # x,y,z [m] 
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        # * initial conditiosn for reset_to_range
        dof_pos_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        dof_vel_range = {'joint_a': [-1., 1.],
                         'joint_b': [-1., 1.]}
        root_pos_range = [[0., 0.],  # x
                          [0., 0.],  # y
                          [0.5, 0.75],  # z
                          [0., 0.],  # roll
                          [0., 0.],  # pitch
                          [0., 0.]]  # yaw
        root_vel_range = [[-0.1, 0.1],  # x
                          [-0.1, 0.1],  # y
                          [-0.1, 0.1],  # z
                          [-0.1, 0.1],  # roll
                          [-0.1, 0.1],  # pitch
                          [-0.1, 0.1]]  # yaw

    class control:
        # 제어방식. target joint angle를 출력
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # actuation scale: target angle = actuationScale * actuation + defaultAngle
        # actuation: 신경망이 출력하는 값, 보통 -1 ~ 1 사이
        # actuationScale: 신경망 출력값을 실제 관절 각도로 바꾸는 스케일링 값
        actuation_scale = 0.5
        # decimation: Number of control actuation updates @ sim DT per policy DT
        # policy DT(Agent가 행동을 선택하는 주기) (정책 주기)가 sim DT의 4배, 에이전트는 4 sim step마다 1번 행동을 선택
        # 신경망 입력을 제어주기와 동일하게 하면 매우 빠른 속도로 신경망을 통과시켜야 해서 학습 속도가 매우 느려짐
        decimation = 4 # 250Hz 4step 당 신경망 1번 실행
        # exp_avg_decay = 0.9 : 90% 이전 액션 + 10% 새로운 액션 => 액션을 부드럽게 만듦
        # exp_avg_decay = 0.1 : 10% 이전 액션 + 90% 새로운 액션 => 액션이 더 빠르게 변함
        exp_avg_decay = None

    class asset:
        file = ""  # path to urdf file
        keypoints = []
        end_effectors = []
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = [] # list of body names contacts with the environment
        terminate_after_contacts_on = [] # 땅에 닿으면 에피소드를 종료할 바디 리스트
        disable_gravity = False
        disable_actuations = False # True: 모든 액추에이터 비활성화(제어신호 무시해서 제자리 유지를 위한 힘만 들어감)
        disable_motors = False # True: 모든 모터 비활성화(인형처럼 풀썩)
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        # 3: P gain(stiffness) and D gain(damping) are applied in the simulation
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up <- 3D 모델링 프로그램에서 y-up로 모델링된 메쉬를 z-up으로 변환

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01
        rotor_inertia = []

    class domain_rand:
        randomize_friction = False    # 마찰력을 무작위로 바꿀지 여부
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = False           # 에피소드 도중 로봇에 외력 적용 여부
        push_interval_s = 15
        max_push_vel_xy = 1.

    # reward settings
    class rewards:
        class weights:
            tracking_lin_vel = 0. # 주어진 속도 명령을 잘 따랐는지
            tracking_ang_vel = 0.
            lin_vel_z = 0.
            ang_vel_xy = 0.
            orientation = 0. # 로봇이 넘어지지 않고 자세를 잘 유지했는지
            torques = 0. # 토크를 너무 많이 쓰지 않았는지
            dof_vel = 0.
            base_height = 0.
            feet_air_time = 0.
            collision = 0. # 충돌하지 않았는지
            feet_stumble = 0. # 발이 비틀리거나 넘어지지 않았는지
            actuation_rate = 0. 
            actuation_rate2 = 0. 
            stand_still = 0.
            dof_pos_limits = 0.
        
        class termination_weights:
            termination = 0. # 로봇이 넘어져 에피소드가 실패하는 순간 부여되는 페널티 가중치

        curriculum = False # curriculum 학습, 쉬운지형 -> 어려운지형
        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.  # ! may want to turn this off
        base_height_target = 1.
        max_contact_force = 100. # forces above this value are penalized
    
    class scaling:
        base_lin_vel = 2.0
        base_ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05

        commands = 1
        # Action scales
        dof_pos = 1
        dof_pos_target = dof_pos  # scale by range of motion
        clip_actions = 100.

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [1, 0, 4]  # [m]
        lookat = [2., 5, 1.]  # [m]
        record = False

    class sim:
        dt =  0.001
        substeps = 1
        gravity = [0., 0. , -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 10.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

# RL Agent and PPO algorithm config
class LeggedRobotRunnerCfg(BaseConfig):
    seed = 2
    runner_class_name = 'OnPolicyRunner'

    class logging:
        enable_local_saving = True
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]     # hidden layer sizes for actor network
        critic_hidden_dims = [512, 256, 128]    # hidden layer sizes for critic network
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        actor_obs = ["observation_a",
                     "observation_b",
                     "these_need_to_be_atributes_(states)_of_the_robot_env"]  # actor가 받는 관측값의 이름 리스트

        critic_obs = ["observation_x",
                      "observation_y",
                      "critic_obs_can_be_the_same_or_different_than_actor_obs"] # critic이 받는 관측값의 이름 리스트

        actions = ["q_des"]  # actor가 출력하는 action의 이름 (control_type='P'였으니 q_des가 출력)
        class noise:
            dof_pos = 0.01
            dof_vel = 0.01
            base_lin_vel = 0.1
            base_ang_vel = 0.2
            projected_gravity = 0.05
            height_measurements = 0.1

        class reward:
            class weights:
                tracking_lin_vel = .0
                tracking_ang_vel = 0.
                lin_vel_z = 0
                ang_vel_xy = 0.
                orientation = 0.
                torques = 0.
                dof_vel = 0.
                base_height = 0.
                collision = 0.
                actuator_rate = 0.
                actuator_rate2 = 0.
                stand_still = 0.
                dof_pos_limits = 0.
            class termination_weights:
                termination = 0.

    class algorithm:
        # PPO algorithm parameters
        class PPO:
            # training params
            value_loss_coef = 1.0 # Actor 손실과 Critic 손실을 합칠 때의 가중치
            use_clipped_value_loss = True
            clip_param = 0.2 # PPO 클리핑 파라미터
            entropy_coef = 0.01 # 엔트로피 보너스 가중치 (높으면 더 탐험을 함)
            num_learning_epochs = 5 # 수집된 데이터를 5번 재사용해서 학습
            num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
            learning_rate = 1.e-4 # 5.e-4
            schedule = 'adaptive' # could be adaptive, fixed
            gamma = 0.99 # 할인율 (미래의 보상을 얼마나 중요하게 생각할지)
            lam = 0.95 # GAE(Generalized Advantage Estimation)의 람다 값
            desired_kl = 0.01
            max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic' # 사용할 정책 모델 클래스
        algorithm_class_name = 'PPO' # 사용할 학습 알고리즘 클래스
        num_steps_per_env = 24 # per iteration (n_steps in Rudin 2021 paper - batch_size = n_steps * n_robots) 
                               # 한 번의 정책 업데이트를 위해 각 환경에서 24step을 수집
        max_iterations = 1500 # number of policy updates 
        SE_learner = None
        # logging
        save_interval = 50 # check for potential saves every this many iterations
        run_name = ''
        experiment_name = 'legged_robot'

        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

#机器人参数：电机减速比30 最高效率点速度95rpm，大腿长0.12m，小腿长0.12m 正常站立高度0.185m

class Htdw4438Cfg(LeggedRobotCfg):
    # ==========================
    # 1. 环境与地形 (适配 HimLoco)
    # ==========================
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        # 角速度(3) + 重力向量(3) + 指令(3) + 关节位置(12) + 关节速度(12) + 上一帧动作(12) = 45
        num_one_step_observations = 45
        
        # [历史信息] 输入过去 6 帧的观测，用于让网络推断环境参数和自身状态
        num_observations = num_one_step_observations * 6 
        
        # 特权观测 (用于 Teacher Policy)
        # 包含：单步观测 + 线速度(3) + 外力(3) + 地形点(187) 
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs * 1
        
        episode_length_s = 20 # 每个回合 20 秒，超时重置

    class terrain:
        mesh_type = 'plane' #'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)# 地形网格中的行数，对应训练中的难度等级，difficulty 随行数递增
        # num_cols=20 时，每列占 5% 的选择权重（1/20=0.05），与 terrain_proportions=[0.1, 0.2, 0.3, 0.3, 0.1] 配合实现地形分布
        num_cols = 20 # number of terrain cols (types)# 地形网格中的列数，对应地形类型的多样性
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.2, 0.3, 0.3, 0.1]
        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces

    # ==========================
    # 2. 初始状态
    # ==========================
    class init_state(LeggedRobotCfg.init_state):
        # 修改：初始高度设为 0.15m (与目标高度一致)，避免落地冲击或奇异点
        pos = [0.0, 0.0, 0.15] 
        default_joint_angles = { 
            'fl_hip_joint': 0,  'fl_thigh_joint': 0.9, 'fl_calf_joint': -1.5,
            'fr_hip_joint': 0, 'fr_thigh_joint': 0.9, 'fr_calf_joint': -1.5,
            'rl_hip_joint': 0,  'rl_thigh_joint': 0.9, 'rl_calf_joint': -1.5,
            'rr_hip_joint': 0, 'rr_thigh_joint': 0.9, 'rr_calf_joint': -1.5,
        }

    # ==========================
    # 3. 控制参数 (适配 HTDW-4438)
    # ==========================
    class control(LeggedRobotCfg.control):
        control_type = 'P'          # 位置控制 (PD Controller)
        stiffness = {'joint': 10.0}   # PD参数保留
        damping = {'joint': 0.3}     
        action_scale = 0.25          # 动作缩放：网络输出通常在 [-1, 1]，乘上 0.25 后变为目标关节弧度 注意请和Deploy后的统一
        decimation = 2              # 控制频率设置 物理引擎 dt = 0.005 (200Hz) -> 控制频率 = 200 / 2 = 100Hz

    # ==========================
    # 4. 指令范围 (低速限制)   
    # ==========================
    class commands(LeggedRobotCfg.commands):
        curriculum = True # 开启课程学习
        max_curriculum = 1.0
        num_commands = 4  # x vel, y vel, yaw vel, heading
        resampling_time = 10. # 每 10 秒重新采样一次指令
        heading_command = True # 是否使用朝向指令
        class ranges(LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-4.0, 4.0] # min max [m/s]
                lin_vel_y = [-2.0, 2.0]   # min max [m/s]
                ang_vel_yaw = [-1.56, 1.56]    # min max [rad/s]
                heading = [-3.14, 3.14]

    # ==========================
    # 5. 资产配置
    # ==========================
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/htdw_4438/urdf/htdw_4438.urdf'
        name = "htdw_4438"
        foot_name = "foot"
        penalize_contacts_on = ["hip", "thigh", "calf", "base"] # 碰撞惩罚：除了脚以外的所有部分碰撞都给予惩罚
        terminate_after_contacts_on = ["base"] # 终止条件：如果机身触地（摔倒），重置回合
        self_collisions = 1 # 自碰撞
        flip_visual_attachments = False 
        
        density = 0.001
        angular_damping = 0. # 移除默认阻尼，完全靠 PD 控制
        linear_damping = 0.
        max_angular_velocity = 9.5
        max_linear_velocity = 20.
        armature = 0.005 # 电机转子惯量 (Armature)

    # ==========================
    # 6. 域随机化 
    # ==========================
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25] # 地面摩擦力随机范围

        randomize_base_mass = False
        added_mass_range = [-0.1, 0.3] # 负载范围
        
        push_robots = True # 随机推力，训练抗干扰
        push_interval_s = 15
        max_push_vel_xy = 0.5
        
        # randomize_motor_offset = True
        # motor_offset_range = [-0.05, 0.05] # 模拟电机零点偏移误差 (±5度)
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1] # 模拟电机力矩输出误差 (±10%)
        
        # 关节 PD 参数随机化
        randomize_kp = True
        kp_range = [0.8, 1.2]
        randomize_kd = True
        kd_range = [0.8, 1.2]
        
        # 外力干扰 (Disturbance)
        disturbance = True
        disturbance_range = [-2.0, 2.0] 
        disturbance_interval = 8
        
        # 延迟随机化 (模拟通信/计算延迟)
        delay = True

    # ==========================
    # 7. 奖励函数
    # ==========================
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.185
        clearance_height_target = -0.135  # 目标脚部离地高度（身体坐标系） 身体高度+clearance_height_target=抬高
    
        class scales(LeggedRobotCfg.rewards.scales):

            termination = -0.0              # 终止条件惩罚
            tracking_lin_vel = 1.0          # 追踪线速度奖励
            tracking_ang_vel = 0.5          # 追踪角速度奖励
            lin_vel_z = -2.0                # 垂直速度惩罚（防止机器人向上跳）
            ang_vel_xy = -0.05              # 水平角速度惩罚（保持姿态稳定）
            orientation = -1.3             # 机身方向惩罚（保持机身水平）
            dof_acc = -2.5e-7               # 关节加速度惩罚（平滑运动）
            joint_power = -2e-5             # 关节功率惩罚（节省能量）

            base_height = -1.0
            # base_height_linear = -1.0        # 线性高度惩罚（保持目标高度）
            default_pos_linear = -0.02        # 默认位置线性惩罚（回归初始姿态）
            diagonal_sync = -0.1             # 对角线腿部同步惩罚（协调性）
            hip_mirror_symmetry = -0.1      # 髋关节镜像对称惩罚

            foot_clearance = -0.0          # 脚部高度惩罚（防止拖脚）
            action_rate = -0.01             # 动作变化率惩罚（平滑控制）
            smoothness = -0.01              # 平滑度惩罚（流畅运动）
            feet_air_time = 0.05             # 脚离地时间奖励（鼓励摆动腿抬起）
            feet_stumble = -0.0             # 脚绊倒惩罚（暂不使用）
            stand_still = -1.0               # 静止状态惩罚
            torques = -0.0                  # 扭矩惩罚（暂不使用）
            dof_vel = -0.0                  # 关节速度惩罚（暂不使用）
            dof_pos_limits = -0.0           # 关节位置限制惩罚（暂不使用）
            dof_vel_limits = -0.0           # 关节速度限制惩罚（暂不使用）
            torque_limits = -0.0            # 扭矩限制惩罚（暂不使用）
            collision = -1.0                # 碰撞惩罚（非脚部分碰撞给予惩罚）

    # ==========================
    # 8. 归一化
    # ==========================
    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    # ==========================
    # 9. 噪声
    # ==========================
    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 
        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01    # 关节编码器噪声
            dof_vel = 0.1     # 速度计算噪声 (通常比较大)
            lin_vel = 0.1     # IMU/状态估计噪声
            ang_vel = 0.2     # 陀螺仪噪声
            gravity = 0.05    # 重力感应噪声
            height_measurements = 0.1 # 高度图噪声
    # ==========================
    # 10. 物理引擎
    # ==========================_reward_soft_dof_pos_limit
    class sim(LeggedRobotCfg.sim):
        dt = 0.005 # 物理仿真步长 5ms 关系到上位机频率
        substeps = 1 # 在每个 dt 内部，物理引擎还要细分几步来算
        gravity = [0., 0., -9.81] # 重力加速度
        up_axis = 1 # Z轴为上方向
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
# 训练参数 (PPO Runner)
# ==========================
            solver_type = 1 # 0: PGS, 1: TGS
            num_position_iterations = 4 # 次数越高，关节连接越紧密，不容易“脱臼”。
            num_velocity_iterations = 1 # 求解器计算“速度误差”的次数。主要解决“碰撞后的反弹速度”和“摩擦力”
            contact_offset = 0.01 # 碰撞检测距离
            rest_offset = 0.0 # 物体静止时的平衡距离
            bounce_threshold_velocity = 0.5 # 反弹阈值。如果撞击速度小于 0.5m/s，就不发生反弹（完全非弹性碰撞）
            max_depenetration_velocity = 1.0 # 最大去穿模速度。如果物体不幸穿模了（嵌进地里），引擎会把它推出来。这个参数限制推出来的最大速度。
            default_buffer_size_multiplier = 5
            max_gpu_contact_pairs = 2**23 # 预分配显存缓冲区的大小

# ==========================
# 训练参数 (PPO Runner)
# ==========================
class Htdw4438CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1e-3 # 降低一点学习率

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'htdw_4438_himloco_v1.1'
        experiment_name = 'flat_htdw_4438'
        max_iterations = 300000 # 最大训练迭代次数
        save_interval = 200 # 每 300 次迭代保存一次模型

        # HimLoco核心配置
        policy_class_name = 'HIMActorCritic' 
        algorithm_class_name = 'HIMPPO'    
        num_steps_per_env = 48



# tensorboard --logdir .

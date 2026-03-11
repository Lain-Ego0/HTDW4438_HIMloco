import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque
import glob
import onnxruntime as ort
from legged_gym.envs import *
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch

import rospy
from sensor_msgs.msg import Joy
import os

import time

default_dof_pos=[0,0.9,-1.8 ,0,0.9,-1.8, 0,0.9,-1.8, 0,0.9,-1.8]#默认角度需要与isacc一致

joy_cmd = [0.0, 0.0, 0.0]

def joy_callback(joy_msg):
    global joy_cmd
    joy_cmd[0] =  joy_msg.axes[1]
    joy_cmd[1] =  joy_msg.axes[0]
    joy_cmd[2] =  joy_msg.axes[3]  # 横向操作

def quat_rotate_inverse(q, v):
    # 确保输入为numpy数组
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    vel = data.sensor('linear-velocity').data.astype(np.double)
    return (q, dq, quat,omega,vel)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd


def resolve_onnx_model_path():
    env_path = os.environ.get("HTDW_ONNX_PATH")
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    default_path = os.path.join(LEGGED_GYM_ROOT_DIR, "onnx", "legged1.onnx")
    if os.path.exists(default_path):
        return default_path

    candidates = sorted(glob.glob(os.path.join(LEGGED_GYM_ROOT_DIR, "onnx", "*.onnx")))
    if candidates:
        return candidates[-1]

    return default_path


class Sim2simCfg(A1RoughCfg):

    class sim_config:
        # print("{LEGGED_GYM_ROOT_DIR}",{LEGGED_GYM_ROOT_DIR})

        mujoco_model_path = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "resources",
            "robots",
            "htdw_4438",
            "xml",
            "scene.xml",
        )  # mujoco模型路径
        
        sim_duration = 60.0
        dt = 0.005 #1Khz底层
        decimation = 4 # 50Hz

    class robot_config:

        kps = np.array(20, dtype=np.double)#PD和isacc内部一致
        kds = np.array(0.5, dtype=np.double)
        tau_limit = 20. * np.ones(12, dtype=np.double)#nm

if __name__ == '__main__':
    rospy.init_node('play')
    rospy.Subscriber('/joy', Joy, joy_callback, queue_size=10)

    policy_model_path = resolve_onnx_model_path()

    policy = ort.InferenceSession(policy_model_path, 
                            providers=['CPUExecutionProvider'])
    model = mujoco.MjModel.from_xml_path(Sim2simCfg.sim_config.mujoco_model_path)#载入初始化位置由XML决定
    model.opt.timestep = Sim2simCfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    model.opt.gravity = (0, 0, -9.81) 
    viewer = mujoco_viewer.MujocoViewer(model, data)

    target_q = np.zeros((12), dtype=np.double)
    action = np.zeros((12), dtype=np.double)
    action_flt = np.zeros((12), dtype=np.double)
    last_actions = np.zeros((12), dtype=np.double)
    lag_buffer = [np.zeros_like(action) for i in range(2+1)]

    hist_obs = deque()
    for _ in range(15):
        hist_obs.append(np.zeros([1,45], dtype=np.double))
    count_lowlevel = 0

    for _ in tqdm(range(int(Sim2simCfg.sim_config.sim_duration*10/ Sim2simCfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat,omega,vel= get_obs(data)#从mujoco获取仿真数据
        q = q[-12:]
        dq = dq[-12:]
        if 1:
            # 1000hz ->50hz
            if count_lowlevel % Sim2simCfg.sim_config.decimation == 0:

                obs = np.zeros([1, 45], dtype=np.float32) #1,45           
                gravity_vec =  np.array([0., 0., -1.], dtype=np.float32)
                proj_gravity = quat_rotate_inverse(quat,gravity_vec)
                obs[0, 0] = omega[0] *Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 1] = omega[1] *Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 2] = omega[2] *Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 3] = proj_gravity[0] 
                obs[0, 4] = proj_gravity[1] 
                obs[0, 5] = proj_gravity[2] 
                obs[0, 6] = (joy_cmd[0])* Sim2simCfg.normalization.obs_scales.lin_vel
                obs[0, 7] = (joy_cmd[1])* Sim2simCfg.normalization.obs_scales.lin_vel
                obs[0, 8] = (joy_cmd[2])* Sim2simCfg.normalization.obs_scales.ang_vel
                obs[0, 9:21] = (q-default_dof_pos) * Sim2simCfg.normalization.obs_scales.dof_pos #g关节角度顺序依据修改为样机
                obs[0, 21:33] = dq * Sim2simCfg.normalization.obs_scales.dof_vel
                obs[0, 33:45] = last_actions#上次控制指令
                obs = np.clip(obs, -Sim2simCfg.normalization.clip_observations, Sim2simCfg.normalization.clip_observations)

                n_proprio=45
                history_len=15
                num_z_encoder = 32

                policy_input = np.zeros([1, n_proprio], dtype=np.float32) 

                for i in range(n_proprio):
                    policy_input[0, i] = obs[0][i]

                policy_output_name = policy.get_outputs()[0].name
                policy_input_name = policy.get_inputs()[0].name

                action[:] = policy.run([policy_output_name], {policy_input_name: policy_input})[0]

                action = np.clip(action, -10,10)

                last_actions=action

                action_flt = action *0.25

                joint_pos_target = action_flt + default_dof_pos
                target_q=joint_pos_target

                target_dq = np.zeros((12), dtype=np.double)
                # Generate PD control
                tau = pd_control(target_q, q, Sim2simCfg.robot_config.kps,
                                target_dq, dq, Sim2simCfg.robot_config.kds)  # Calc torques
                tau = np.clip(tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit)  # Clamp torques
                # torques = np.clip(tau, -torque_limits, torque_limits)
                data.ctrl = tau
            time.sleep(0.003)
            mujoco.mj_step(model, data)
            
        else:
            target_q = default_dof_pos
            target_dq = np.zeros((12), dtype=np.double)
            # Generate PD control
            tau = pd_control(target_q, q, Sim2simCfg.robot_config.kps,
                            target_dq, dq, Sim2simCfg.robot_config.kds)  # Calc torques
            tau = np.clip(tau, -Sim2simCfg.robot_config.tau_limit, Sim2simCfg.robot_config.tau_limit)  # Clamp torques
            data.ctrl = tau

            mujoco.mj_step(model, data)

        viewer.render()
        count_lowlevel += 1

    viewer.close()

    

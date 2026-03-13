import time
import os
import argparse
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import yaml
from collections import deque
from pynput import keyboard
from onnx_path_utils import resolve_onnx_path

# ================= 1. 路径配置 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
YAML_PATH = os.path.join(SCRIPT_DIR, "configs", "htdw_4438.yaml")
XML_PATH = os.path.join(PROJECT_ROOT, "resources", "robots", "htdw_4438", "xml", "scene.xml")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy HTDW4438 policy in MuJoCo.")
    parser.add_argument("--onnx", type=str, default=None, help="Absolute or relative path to ONNX policy.")
    return parser.parse_args()


ARGS = parse_args()
ONNX_PATH = resolve_onnx_path(
    project_root=PROJECT_ROOT,
    cli_onnx=ARGS.onnx,
    env_vars=["HTDW_ONNX_PATH"],
    onnx_glob="flat_htdw_4438*.onnx",
    robot_name="htdw_4438",
)

# 打印路径信息，便于调试路径是否正确
print(f"YAML: {YAML_PATH}")
print(f"XML : {XML_PATH}")
print(f"ONNX: {ONNX_PATH}")

# ================= 2. 全局变量 =================
# [vx, vy, omega]
cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32) 
paused = False
default_dof_pos = None # 将在 main 中加载

# ================= 3. 辅助函数 =================

def quat_rotate_inverse(q, v):
    """
    参考 deploy_lanbu.py: 计算向量 v 在四元数 q 表示的坐标系下的逆旋转
    用于将重力向量转到机身坐标系
    q: [w, x, y, z] (MuJoCo 格式)
    v: [x, y, z]
    """
    q_w = q[0]
    q_vec = q[1:4]
    
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def build_policy_input(obs_raw, history_buffer, input_dim, num_obs):
    if input_dim == num_obs:
        history_buffer.appendleft(obs_raw.copy())
        return np.concatenate(list(history_buffer), axis=0).reshape(1, -1)
    if input_dim == 64:
        policy_input = np.zeros((1, 64), dtype=np.float32)
        policy_input[0, :45] = obs_raw
        return policy_input
    if input_dim == 45:
        return obs_raw.reshape(1, -1)
    raise ValueError(f"Unsupported ONNX input dim: {input_dim}")

# def pd_control(target_q, q, kp, kd, qvel, kds_val):
#     # 修改：确保传入并使用了微分增益 kds_val
#     return kp * (target_q - q) - kds_val * qvel

# ================= 4. 键盘控制 =================
def on_press(key):
    global cmd
    try:
        if key == keyboard.Key.up:
            cmd[0] = 0.6  # 前进
        elif key == keyboard.Key.down:
            cmd[0] = -0.4 # 后退
        elif key == keyboard.Key.left:
            cmd[2] = 0.8  # 左转
        elif key == keyboard.Key.right:
            cmd[2] = -0.8 # 右转
    except AttributeError:
        pass

def on_release(key):
    global cmd
    try:
        if key == keyboard.Key.up or key == keyboard.Key.down:
            cmd[0] = 0.0
        elif key == keyboard.Key.left or key == keyboard.Key.right:
            cmd[2] = 0.0
    except AttributeError:
        pass

def key_callback(keycode):
    global paused
    if chr(keycode) == ' ':
        paused = not paused
        print(f"Paused: {paused}")

# ================= 5. 主程序 =================
def run_simulation():
    global cmd, default_dof_pos
    
    # --- 加载 YAML 配置 ---
    if not os.path.exists(YAML_PATH):
        print(f"错误: 找不到配置文件 {YAML_PATH}")
        return

    with open(YAML_PATH, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 提取参数
    sim_dt = float(config.get('simulation_dt', 0.005))
    control_decimation = int(config.get('control_decimation', 2))
    num_obs = int(config.get('num_obs', 270))
    num_one_step_obs = int(config.get('num_one_step_obs', 45))
    init_base_height = float(config.get('init_base_height', 0.15))
    
    kps = np.array(config['kps'], dtype=np.float32)
    kds = np.array(config['kds'], dtype=np.float32)
    default_dof_pos = np.array(config['default_angles'], dtype=np.float32)
    
    # 缩放因子
    lin_vel_scale = config['lin_vel_scale']
    ang_vel_scale = config['ang_vel_scale']
    dof_pos_scale = config['dof_pos_scale']
    dof_vel_scale = config['dof_vel_scale']
    action_scale = config['action_scale']
    cmd_scale = np.array(config['cmd_scale'], dtype=np.float32)

    # --- 加载 MuJoCo & ONNX ---
    if not os.path.exists(XML_PATH):
        print(f"错误: 找不到模型文件 {XML_PATH}")
        return
        
    print("正在加载 MuJoCo 模型...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = sim_dt
    use_gyro_sensor = True
    try:
        _ = data.sensor("angular-velocity").data
    except KeyError:
        use_gyro_sensor = False
        print("警告: 未找到传感器 'angular-velocity'，将回退到 data.qvel[3:6]。")

    # print(f"正在加载 ONNX: {ONNX_PATH}")
    # ort_session = ort.InferenceSession(ONNX_PATH)
    # input_name = ort_session.get_inputs()[0].name

    print(f"正在加载 ONNX: {ONNX_PATH}")
    ort_session = ort.InferenceSession(ONNX_PATH)
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    input_dim = int(input_shape[-1]) if isinstance(input_shape[-1], int) else num_obs
    print(f"ONNX Input Shape: {input_shape}")

    # --- 初始化状态 ---
    data.qpos[7:] = default_dof_pos
    data.qpos[2] = init_base_height # 起始高度（与训练 cfg.init_state.pos[2] 对齐）
    mujoco.mj_forward(model, data)
    
    target_dof_pos = default_dof_pos.copy()
    action = np.zeros(12, dtype=np.float32)
    
    # 键盘监听
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    print("仿真开始！使用方向键控制移动，空格键暂停。")

    history_len = max(1, num_obs // num_one_step_obs)
    obs_dim = num_one_step_obs
    obs_history_buffer = deque([np.zeros(obs_dim, dtype=np.float32) for _ in range(history_len)], maxlen=history_len)

    # --- 仿真循环 ---
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        step_counter = 0
        while viewer.is_running():
            step_start = time.time()
            
            if not paused:
                # ================= 策略控制 (100Hz) =================
                if step_counter % control_decimation == 0:
                    # 1. 获取传感器数据
                    qj = data.qpos[7:]
                    dqj = data.qvel[6:]
                    quat = data.qpos[3:7] # [w, x, y, z]
                    if use_gyro_sensor:
                        omega = data.sensor("angular-velocity").data.astype(np.float32) # 机身角速度（IMU）
                    else:
                        omega = data.qvel[3:6].astype(np.float32) # 兜底：机身角速度
                    # omega = data.qvel[3:6] # 机身角速度
                    
                    # 2. 数据转换与归一化
                    gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
                    proj_gravity = quat_rotate_inverse(quat, gravity_vec)
                    
                    qj_norm = (qj - default_dof_pos) * dof_pos_scale
                    dqj_norm = dqj * dof_vel_scale
                    omega_norm = omega * ang_vel_scale
                    cmd_norm = cmd * cmd_scale
                    
                    # 3. 构建观测向量 (45维)
                    # 顺序与训练侧 LeggedRobot.compute_observations 保持一致：
                    # Cmd(3) + AngVel(3) + Gravity(3) + DofPos(12) + DofVel(12) + LastAction(12)
                    obs_list = []
                    obs_list.extend(cmd_norm)        # 0-2
                    obs_list.extend(omega_norm)      # 3-5
                    obs_list.extend(proj_gravity)    # 6-8
                    obs_list.extend(qj_norm)         # 9-20
                    obs_list.extend(dqj_norm)        # 21-32
                    obs_list.extend(action)          # 33-44
                    
                    obs_raw = np.array(obs_list, dtype=np.float32)
                    
                    policy_input = build_policy_input(
                        obs_raw=obs_raw,
                        history_buffer=obs_history_buffer,
                        input_dim=input_dim,
                        num_obs=num_obs,
                    )
                    
                    # 5. 推理
                    outputs = ort_session.run(None, {input_name: policy_input})
                    raw_action = outputs[0][0]
                    
                    # 6. 处理输出
                    raw_action = np.clip(raw_action, -10, 10)
                    action = raw_action # 更新 LastAction
                    target_dof_pos = raw_action * action_scale + default_dof_pos

                # ================= 物理执行 (500Hz) =================
                tau = pd_control(target_dof_pos, data.qpos[7:], kps, 
                                 np.zeros_like(kds), data.qvel[6:], kds)
                # 与模型 actuator 的控制范围对齐，避免硬编码饱和范围与训练资产不一致
                tau_limit = np.abs(model.actuator_ctrlrange[:, 1])
                tau = np.clip(tau, -tau_limit, tau_limit)
                data.ctrl[:] = tau
                
                mujoco.mj_step(model, data)
                step_counter += 1

            viewer.sync()
            
            # 帧率同步
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_simulation()

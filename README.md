# HTDW4438_HIMloco 🤖



tensorboard --logdir .

export PYTHONPATH=$PWD
python legged_gym/scripts/train.py --task=htdw_4438 --headless

python legged_gym/scripts/train.py --task=opendoge --headless
export PYTHONPATH=. && python legged_gym/scripts/play.py --task=opendoge --load_run Mar10_17-53-37_htdw_4438_himloco_v1.1 --checkpoint 2000

export PYTHONPATH=. && python legged_gym/scripts/play.py --task=htdw_4438 --load_run Mar10_17-53-37_htdw_4438_himloco_v1.1 --checkpoint 2000


conda run --no-capture-output -n lain_rl python deploy/deploy_mujoco/deploy_opendoge.py --onnx /home/sunteng/Documents/GitHub/HTDW4438_HIMloco/onnx/flat_opendoge_20260312_163410_model_600.onnx


[![Hardware](https://img.shields.io/badge/Hardware-HTDW4438--OpenDog-blue)](https://github.com/Lain-Ego0/HTDW4438-OpenDog)

本项目是为 **[HTDW4438-OpenDog](https://github.com/Lain-Ego0/HTDW4438-OpenDog)** 四足机器人的强化学习训练框架。

> **致谢**：
> 本项目衍生自 [HIMLoco](https://github.com/InternRobotics/HIMLoco)。
> 同时参考了：
> - Github
> [quadruped_rl](https://github.com/Benxiaogu/quadruped_rl)
> [himloco_lab](https://github.com/IsaacZH/himloco_lab)
> - bilibili
>[himloco+stage](https://www.bilibili.com/video/BV1Ut2fBwEUK/?spm_id_from=333.337.search-card.all.click&vd_source=193a56b6f00b33090010ba20d05cfef7)
[【四足运控，从入门到精通】第二期 | 训练代码解释，奖励函数设计，仿真环境配置](https://www.bilibili.com/video/BV1Mv4vzhELA/?spm_id_from=333.337.search-card.all.click&vd_source=193a56b6f00b33090010ba20d05cfef7)

![HTDW4438-OpenDog](./assets/image.png)

## 📦 安装配置

系统版本：Ubuntu20.04 LTS

- 打开终端，运行以下命令：

1.安装CUDA

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

2.安装虚拟环境并直接导入环境包

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

```bash
conda env create -f Lain_env.yml
```
> **提示**：这个过程会下载 PyTorch、TensorFlow 和 CUDA 相关的包，耗时较长（取决于网速）。如果下载太慢，建议先配置 Conda 国内镜像源。

- 安装完成后，使用以下命令进入环境：

```bash
conda activate Lain_env
```
- 验证安装

```bash
python -c "import torch; import tensorflow as tf; print('PyTorch GPU:', torch.cuda.is_available()); print('TF GPU:', len(tf.config.list_physical_devices('GPU')) > 0)"
```


## 🛠️ 使用指南

### 1. Train

```bash
python legged_gym/scripts/train.py --task=任务1 --num_envs=256 --rl_device=cuda:0
# 从上次运行中继续训练
python legged_gym/scripts/train.py --task=任务1 --resume --load_run 20250623_good --checkpoint 47000
```

#### ⚙️  参数说明
- `--task`: 必选参数，值可选(galileo, zsl1, go2, g1, h1, h1_2)
- `--headless`: 默认启动图形界面，设为 true 时不渲染图形界面（效率更高）
- `--resume`: 从日志中选择 checkpoint 继续训练
- `--experiment_name`: 运行/加载的 experiment 名称
- `--run_name`: 运行/加载的 run 名称
- `--load_run`: 加载运行的名称，默认加载最后一次运行
- `--checkpoint`: checkpoint 编号，默认加载最新一次文件
- `--num_envs`: 并行训练的环境个数
- `--seed`: 随机种子
- `--max_iterations`: 训练的最大迭代次数
- `--sim_device`: 仿真计算设备，指定 CPU 为 `--sim_device=cpu`
- `--rl_device`: 强化学习计算设备，指定 CPU 为 `--rl_device=cpu`

**默认保存训练结果**：`logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`


### 2. Play


```bash
python legged_gym/scripts/play.py --task=galileo
python legged_gym/scripts/play.py --task=galileo --load_run 20250623_good --checkpoint 47000
# tensorboard
pip install tensorflow
tensorboard --logdir=./logs/……
```

#### ⚙️  参数说明

- Play 启动参数与 Train 相同。
- 默认加载实验文件夹上次运行的最后一个模型。
- 可通过 `load_run` 和 `checkpoint` 指定其他模型。

#### 💾 导出网络

Play 会导出 Actor 网络，保存于 `logs/{experiment_name}/exported/policies` 中：
- 普通网络（MLP）导出为 `policy_1.pt`
- RNN 网络，导出为 `policy_lstm_1.pt`

### Play Results
---
### 3. Sim2Sim

#### keyboard 操作说明
|前进|后退|左移|右移|左转|右转|
|---|---|---|---|---|---|
|Up Arrow|Down Arrow|Left Arrow|Right Arrow|Ctrl+Left Arrow|Ctrl+Right Arrow|
#### Logitech F710 操作说明
x模式，左边摇杆控制前移/后移/左移/右移，右边摇杆控制左转/右转。

#### ⚙️  参数说明
- `config_name`: 配置文件，默认查询路径为 `deploy/deploy_mujoco/configs/`
#### ➡️  替换网络模型
---

### 4. Sim2Real
#### lcm配置


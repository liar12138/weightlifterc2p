# H2 杠铃 C2P 在线训练

这部分实现第一阶段的 camera-to-proprioception（C2P）训练：机器人由已经训练好的特权教师策略控制，深度相机和机器人本体观测输入 C2P 网络，MuJoCo 仿真中的真实杠铃状态只作为监督标签。教师策略和 C2P 网络使用完全独立的参数，训练过程中不会更新教师。

## 文件

| 文件 | 作用 |
| --- | --- |
| src/tasks/weight_lifter_H2/networks/barbell_c2p.py | 深度历史 CNN、proprio 编码器和 16D 输出头 |
| src/tasks/weight_lifter_H2/c2p_observation.py | 计算机器人坐标系下的 16D 仿真真值 |
| src/tasks/weight_lifter_H2/c2p_runner.py | 在线采样、教师推理、MSE 更新和 checkpoint 保存 |
| src/tasks/weight_lifter_H2/configs/c2p_cfg.py | C2P 环境和 runner 配置 |
| scripts/train_c2p.py | 训练入口 |
| scripts/eval_c2p.py | 在线误差评估入口 |

## 观测和标签

C2P 网络的输入不包含任何真实杠铃字段：

- actor：274 维本体观测；
- depth_camera：深度历史，默认形状为 [B, 10, 27, 48]；
- history_length 可以通过命令行修改。

监督标签组为 c2p_target，形状 [B, 16]，字段顺序固定为：

    0:3    barbell_pos_b                 杠铃质心位置，机器人 body frame
    3:7    barbell_quat_b                杠铃姿态四元数，机器人 body frame
    7:10   bellbar_vector_b              杠铃轴方向，机器人 body frame
    10:13  left_hand_to_plate_b           左手到左杠铃片向量
    13:16  right_hand_to_plate_b          右手到右杠铃片向量

四元数和位置都按照当前仿真坐标系定义，评估时必须使用同样的字段顺序和坐标系。

## CNN 结构

深度历史帧沿通道维输入 CNN：

    T x 27 x 48
    Conv2d(T, 16, kernel=5, stride=2, padding=2)
    ELU + MaxPool2d(2)
    Conv2d(16, 32, kernel=3, stride=2, padding=1)
    ELU
    Conv2d(32, 64, kernel=3, stride=2, padding=1)
    ELU
    Flatten -> Linear(256) -> Linear(128)

274 维 proprio 先经过 Linear(274, 128)、LayerNorm 和 ELU，再与 128 维深度特征拼接，经过 256 -> 128 -> 16 输出头。位置和向量使用固定尺度归一化，四元数不缩放。

## 教师策略和动作文件

默认教师 checkpoint：

    src/assets/model_28300.pt

教师任务：

    Unitree-H2-Pullup-A-Barbell-No-Camera-Privi

默认动作文件：

    src/assets/motion/h2o6/1_h2o6_new_s6.npz

教师加载使用 strict=True。启动时应看到类似输出：

    [C2P] actor=(N, 274) depth=(N, 10, 27, 48) target=(N, 16)

## 训练

进入仓库目录并使用 MJLab 环境：

    cd /home/tcs/8t01/lhx/weightlifterc2p

    /home/tcs/.conda/envs/unitree_rl_mjlab_new/bin/python +      scripts/train_c2p.py +      --num-envs 64 +      --history-length 10 +      --iterations 10001 +      --device cuda:0

常用参数：

    --num-envs              并行环境数，显存不足时调小
    --history-length        深度历史帧数，默认 10
    --num-steps-per-env     每次采样步数，默认 24
    --c2p-epochs            每批数据重复训练轮数，默认 5
    --c2p-mini-batches      mini-batch 数，默认 4
    --c2p-learning-rate     AdamW 学习率，默认 1e-4
    --save-interval          checkpoint 保存间隔，默认 100
    --teacher-checkpoint    教师权重路径
    --log-dir                C2P 输出目录

checkpoint 默认保存在：

    logs/h2_barbell_c2p/c2p_model_<iteration>.pt

checkpoint 只包含 C2P 模型、C2P 优化器和配置，不覆盖教师策略。

## 评估

使用训练得到的 checkpoint 在线运行仿真：

    /home/tcs/.conda/envs/unitree_rl_mjlab_new/bin/python +      scripts/eval_c2p.py +      --checkpoint logs/h2_barbell_c2p/c2p_model_10001.pt +      --num-envs 64 +      --steps 500 +      --history-length 10 +      --device cuda:0

输出指标：

    position_rmse_m       杠铃位置向量 RMSE，单位米
    position_mae_m        杠铃位置平均绝对误差，单位米
    position_p95_m        位置误差 95 分位数
    position_within_5cm   位置误差小于 5 cm 的样本比例
    quaternion_l2_rmse    四元数 L2 误差
    vector_l2_rmse        方向和左右手向量的 L2 误差

建议首先观察位置误差。初始验收目标可以设置为位置 RMSE 小于 5 cm、P95 小于 10 cm，并且超过 90% 样本误差小于 5 cm。

## 推荐验证顺序

1. 先使用 --num-envs 1 --iterations 0 检查四个观测组形状。
2. 使用 --num-steps-per-env 1 --c2p-epochs 1 做单步 smoke test。
3. 在固定数据上重复训练，确认 loss 能持续下降。
4. 再增加环境数和训练迭代数。
5. 使用独立 motion 片段或随机种子运行 eval_c2p.py，不要只在训练采样数据上评估。

## 注意事项

- c2p_target 和 barbell 只用于教师和监督标签，不能加入 C2P 输入。
- 修改 history_length 时，训练和评估必须使用相同数值。
- 如果出现 checkpoint shape mismatch，优先检查教师任务、动作文件和 274 维 actor 观测是否一致。
- 当前脚本是第一阶段监督学习，不包含第二阶段策略蒸馏。

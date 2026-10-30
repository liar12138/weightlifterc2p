# Touch-A-Box（G1 + Revo2 手部 ContactSensor 验证）

不修改现有任务代码 / `src/__init__.py` / conda 环境。本目录为**新建**任务；通过 `scripts/play_touch_a_box.py` 注册并启动。

## 做什么

- 机器人：本地 `unitree_g1_revo2`
- 传感器：单个 `hand_tactile` ContactSensor，覆盖掌心 + 指节 + `touch_link`（约 34 列）
- `secondary=None`：碰什么都计数
- 固定箱子在手下方；初始姿态手臂前伸、手指半握
- 每 0.5s 控制台打印非零接触
- GUI：`viser`（浏览器，适合无桌面服务器）

## 运行

```bash
conda activate capylab
cd weightlifter
python scripts/play_touch_a_box.py --agent zero --viewer viser --num_envs 1
```

说明：未改 `src/__init__.py`（其中部分 combine_* 任务缺失会导致 `from src import *` 失败），因此用本脚本注册任务并启动 viser，而不是直接跑 `scripts/play.py`。

终端会出现 viser 链接；在浏览器里微调视角，同时看控制台 `[hand_tactile]` 输出。

## 期望现象

1. 启动后打印 34 个 `primary_names`
2. 手压到箱子后控制台周期性出现类似：

```text
[hand_tactile] primaries=34 active=K | left_pinky_proximal_link: |F|=... | ...
```

3. 浏览器打开 viser 链接，可看场景/接触；姿态可在 `configs/env_cfgs.py` 的 `_TOUCH_POSE_JOINT_POS` / `_BOX_POS` 微调

无 GUI 冒烟（可选）：

```bash
python scripts/_smoke_touch_a_box.py
```

## 说明

- 未改任何已有任务代码、`src/__init__.py`、conda 环境
- `src/__init__.py` 里缺失的 `combine_*` 包会导致 `from src import *` 失败，因此用 `scripts/play_touch_a_box.py`（命名空间注入 + 注册本任务）而不是直接 `scripts/play.py`
- 传感器：`secondary=None`，`reduce=netforce`，每部位一列力

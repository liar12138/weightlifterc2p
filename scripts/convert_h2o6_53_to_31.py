#!/usr/bin/env python3
"""将 H2 + LinkerHand O6 的 53 维动作文件转换为 31 维（无手）动作文件。

背景
----
H2 + O6 动作文件（h2_o6.xml）的关节顺序为 53 维：
    左腿(6), 右腿(6), 腰(3), 头(2), 左臂(7), 左手(11), 右臂(7), 右手(11)

对应的 body 顺序（NPZ 中不含 world，共 56 个 body）为：
    pelvis(1), 左腿(6), 右腿(6), 腰(2), torso(1), 头(2),
    左臂(7), 左手(12), 右臂(7), 右手(12)

去掉双手信息后，得到 31 维关节（与 h2.xml 的 31 自由度模型一致）：
    左腿(6), 右腿(6), 腰(3), 头(2), 左臂(7), 右臂(7)

以及 32 个 body（同样与 h2.xml 一致）：
    pelvis, 左腿(6), 右腿(6), 腰(2), torso, 头(2), 左臂(7), 右臂(7)

用法
----
    python scripts/convert_h2o6_53_to_31.py <输入.npz> [输出.npz]

示例
----
    python scripts/convert_h2o6_53_to_31.py \
        src/assets/motion/h2/1_h2o6_new_s3.npz \
        src/assets/motion/h2/1_h2o6_new_s3_31.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# 53 维关节中需要保留的索引（去掉左右手各 11 个关节）。
# 0..23 = 左腿(6) + 右腿(6) + 腰(3) + 头(2) + 左臂(7)
# 35..41 = 右臂(7)
JOINT_KEEP_IDX = list(range(0, 24)) + list(range(35, 42))

# 56 个 body 中需要保留的索引（去掉左右手各 12 个 body）。
# 0..24 = pelvis + 左腿(6) + 右腿(6) + 腰(2) + torso + 头(2) + 左臂(7)
# 37..43 = 右臂(7)
BODY_KEEP_IDX = list(range(0, 25)) + list(range(37, 44))

# 期望的输入维度
JOINT_DIM_IN = 53
JOINT_DIM_OUT = 31
BODY_COUNT_IN = 56
BODY_COUNT_OUT = 32


def convert(input_path: Path, output_path: Path) -> None:
    with np.load(input_path) as data:
        # 校验输入维度
        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        body_pos_w = data["body_pos_w"]

        if joint_pos.shape[-1] != JOINT_DIM_IN:
            raise ValueError(
                f"joint_pos 维度为 {joint_pos.shape[-1]}，"
                f"期望 {JOINT_DIM_IN} 维（53 维 H2+O6）"
            )
        if body_pos_w.shape[1] != BODY_COUNT_IN:
            raise ValueError(
                f"body_pos_w 的 body 数量为 {body_pos_w.shape[1]}，"
                f"期望 {BODY_COUNT_IN}（含双手的 56 个 body）"
            )

        print(f"[IN] {input_path}")
        print(f"     joint_pos: {joint_pos.shape}")
        print(f"     joint_vel: {joint_vel.shape}")
        print(f"     body_pos_w: {body_pos_w.shape}")

        out = {
            "joint_pos": joint_pos[:, JOINT_KEEP_IDX],
            "joint_vel": joint_vel[:, JOINT_KEEP_IDX],
            "body_pos_w": body_pos_w[:, BODY_KEEP_IDX],
            "body_quat_w": data["body_quat_w"][:, BODY_KEEP_IDX],
            "body_lin_vel_w": data["body_lin_vel_w"][:, BODY_KEEP_IDX],
            "body_ang_vel_w": data["body_ang_vel_w"][:, BODY_KEEP_IDX],
        }

        # fps 等标量元数据原样保留
        for key in data.files:
            if key not in out:
                out[key] = data[key]

    # 校验输出维度
    assert out["joint_pos"].shape[-1] == JOINT_DIM_OUT
    assert out["joint_vel"].shape[-1] == JOINT_DIM_OUT
    assert out["body_pos_w"].shape[1] == BODY_COUNT_OUT
    assert out["body_quat_w"].shape[1] == BODY_COUNT_OUT
    assert out["body_lin_vel_w"].shape[1] == BODY_COUNT_OUT
    assert out["body_ang_vel_w"].shape[1] == BODY_COUNT_OUT

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **out)

    print(f"[OUT] {output_path}")
    for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                "body_lin_vel_w", "body_ang_vel_w"):
        print(f"      {key}: {out[key].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 H2+O6 的 53 维动作转换为 31 维（去掉双手）"
    )
    parser.add_argument("input", type=Path, help="输入的 53 维 npz 文件")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="输出的 31 维 npz 文件（默认与输入同名并加 _31dof 后缀）",
    )
    args = parser.parse_args()

    input_path: Path = args.input
    if args.output is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_31dof{input_path.suffix}"
        )
    else:
        output_path = args.output

    convert(input_path, output_path)


if __name__ == "__main__":
    main()

import numpy as np

input_path = "/data/weightlifter/src/assets/motion/h2o6/1_h2o6_promax_cut_aligned.npz"
output_path = "/data/weightlifter/src/assets/motion/h2o6/1_h2o6_promax_cut_aligned_from_220.npz"

data = np.load(input_path)

# 用 joint_pos 确定真正的动作帧数
num_frames = data["joint_pos"].shape[0]

print("检测到原始帧数:", num_frames)

new_data = {}

for key in data.files:
    arr = data[key]

    # 逐帧数据
    if (
        arr.ndim > 0
        and arr.shape[0] == num_frames
    ):
        # 删除 220 之前，保留 220 及之后
        new_data[key] = arr[220:]

        print(
            f"[CUT] {key}: "
            f"{arr.shape} -> {new_data[key].shape}"
        )

    else:
        # fps 等元数据保持不变
        new_data[key] = arr

        print(
            f"[KEEP] {key}: {arr.shape}"
        )

np.savez(
    output_path,
    **new_data,
)

print("保存到:", output_path)
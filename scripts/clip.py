import numpy as np

input_path = "/data/weightlifter/src/assets/motion/h2o6/1_h2o6_promax_cut_aligned_0_80.npz"
output_path = "/data/weightlifter/src/assets/motion/h2o6/1_h2o6_promax_cut_aligned_0_80_hold50.npz"

data = np.load(input_path)

num_frames = data["joint_pos"].shape[0]
hold_frames = 50

new_data = {}

for key in data.files:
    arr = data[key]

    if arr.ndim > 0 and arr.shape[0] == num_frames:
        last_frame = arr[-1:]

        repeated = np.repeat(
            last_frame,
            hold_frames,
            axis=0,
        )

        new_data[key] = np.concatenate(
            [arr, repeated],
            axis=0,
        )

        print(
            f"[EXTEND] {key}: "
            f"{arr.shape} -> {new_data[key].shape}"
        )

    else:
        new_data[key] = arr
        print(f"[KEEP] {key}: {arr.shape}")

np.savez(
    output_path,
    **new_data,
)

print("保存完成：", output_path)
import torch


# ==========================================================
# 之前会看相机抓杆的旧模型
# ==========================================================
OLD_MODEL = (
    "/data/weightlifter/logs/g1_tracking/rsl_rl/"
    "2026-08-22_14-09-33/model_32400.pt"
)

# ==========================================================
# 当前已经会蹲下的新模型
# ==========================================================
CURRENT_MODEL = (
    "/data/weightlifter/logs/g1_tracking/rsl_rl/"
    "2026-08-22_14-10-50/model_1100.pt"
)

# ==========================================================
# 输出 hybrid 模型
# ==========================================================
SAVE_MODEL = (
    "/data/weightlifter/logs/g1_tracking/rsl_rl/"
    "2026-08-22_14-10-50/model_convert333.pt"
)


if __name__ == "__main__":

    print("Loading OLD grab model:")
    print(OLD_MODEL)

    old_ckpt = torch.load(
        OLD_MODEL,
        map_location="cpu",
    )

    print("\nLoading CURRENT squat model:")
    print(CURRENT_MODEL)

    current_ckpt = torch.load(
        CURRENT_MODEL,
        map_location="cpu",
    )

    old_actor = old_ckpt["actor_state_dict"]
    current_actor = current_ckpt["actor_state_dict"]

    # ======================================================
    # 1. old CNN -> current CNN
    # ======================================================

    print("\n======================================")
    print("COPY OLD CNN -> CURRENT CNN")
    print("======================================")

    for old_key, old_value in old_actor.items():

        if not old_key.startswith("cnn."):
            continue

        new_key = old_key

        if new_key not in current_actor:
            raise KeyError(
                f"Current actor 不存在 key: {new_key}"
            )

        if old_value.shape != current_actor[new_key].shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"old: {old_key} {tuple(old_value.shape)}\n"
                f"new: {new_key} {tuple(current_actor[new_key].shape)}"
            )

        current_actor[new_key] = old_value.clone()

        print(
            f"[CNN] {old_key:35s}"
            f" -> {new_key:35s}"
            f" {tuple(old_value.shape)}"
        )

    # ======================================================
    # 2. old MLP -> current mlp_tracking
    # ======================================================

    print("\n======================================")
    print("COPY OLD MLP -> CURRENT MLP_TRACKING")
    print("======================================")

    for old_key, old_value in old_actor.items():

        if not old_key.startswith("mlp."):
            continue

        suffix = old_key[len("mlp."):]
        new_key = "mlp_tracking." + suffix

        if new_key not in current_actor:
            raise KeyError(
                f"Current actor 不存在 key: {new_key}"
            )

        if old_value.shape != current_actor[new_key].shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"old: {old_key} {tuple(old_value.shape)}\n"
                f"new: {new_key} {tuple(current_actor[new_key].shape)}"
            )

        current_actor[new_key] = old_value.clone()

        print(
            f"[MLP] {old_key:35s}"
            f" -> {new_key:35s}"
            f" {tuple(old_value.shape)}"
        )

    # ======================================================
    # 3. old obs_normalizer -> current obs_normalizer
    # ======================================================

    print("\n======================================")
    print("COPY OLD OBS_NORMALIZER")
    print("======================================")

    for old_key, old_value in old_actor.items():

        if not old_key.startswith("obs_normalizer."):
            continue

        new_key = old_key

        if new_key not in current_actor:
            raise KeyError(
                f"Current actor 不存在 normalizer key: {new_key}"
            )

        if old_value.shape != current_actor[new_key].shape:
            raise ValueError(
                f"Normalizer shape mismatch:\n"
                f"old: {old_key} {tuple(old_value.shape)}\n"
                f"new: {new_key} {tuple(current_actor[new_key].shape)}"
            )

        current_actor[new_key] = old_value.clone()

        print(
            f"[NORMALIZER] {old_key:35s}"
            f" -> {new_key:35s}"
            f" {tuple(old_value.shape)}"
        )

    # 写回 actor
    current_ckpt["actor_state_dict"] = current_actor

    # ======================================================
    # 4. 其他参数继续保留 model_1100
    # ======================================================

    print("\n======================================")
    print("KEEP FROM CURRENT model_1100.pt")
    print("======================================")

    print("[KEEP] fk_cnn")
    print("[KEEP] mlp_grap")
    print("[KEEP] distribution")
    print("[KEEP] critic")
    print("[KEEP] optimizer")
    print("[KEEP] iter")

    # ======================================================
    # 5. 验证 CNN
    # ======================================================

    print("\n======================================")
    print("VERIFY CNN")
    print("======================================")

    cnn_ok = True

    for old_key, old_value in old_actor.items():

        if not old_key.startswith("cnn."):
            continue

        same = torch.equal(
            old_value,
            current_actor[old_key],
        )

        print(f"{old_key:40s}: {same}")

        if not same:
            cnn_ok = False

    # ======================================================
    # 6. 验证 mlp_tracking
    # ======================================================

    print("\n======================================")
    print("VERIFY MLP_TRACKING")
    print("======================================")

    tracking_ok = True

    for old_key, old_value in old_actor.items():

        if not old_key.startswith("mlp."):
            continue

        suffix = old_key[len("mlp."):]
        new_key = "mlp_tracking." + suffix

        same = torch.equal(
            old_value,
            current_actor[new_key],
        )

        print(
            f"{old_key:30s}"
            f" -> {new_key:40s}: {same}"
        )

        if not same:
            tracking_ok = False

    # ======================================================
    # 7. 验证 obs_normalizer
    # ======================================================

    print("\n======================================")
    print("VERIFY OBS_NORMALIZER")
    print("======================================")

    normalizer_ok = True

    for old_key, old_value in old_actor.items():

        if not old_key.startswith("obs_normalizer."):
            continue

        same = torch.equal(
            old_value,
            current_actor[old_key],
        )

        print(
            f"{old_key:40s}: {same}"
        )

        if not same:
            normalizer_ok = False

    if not cnn_ok:
        raise RuntimeError("CNN 权重验证失败")

    if not tracking_ok:
        raise RuntimeError("mlp_tracking 权重验证失败")

    if not normalizer_ok:
        raise RuntimeError("obs_normalizer 权重验证失败")

    # ======================================================
    # 8. 保存
    # ======================================================

    torch.save(
        current_ckpt,
        SAVE_MODEL,
    )

    print("\n======================================")
    print("转换完成")
    print("======================================")

    print(f"\nSaved to:\n{SAVE_MODEL}")

    # ======================================================
    # 9. 重新 load 检查
    # ======================================================

    check = torch.load(
        SAVE_MODEL,
        map_location="cpu",
    )

    print("\nSaved checkpoint keys:")
    print(check.keys())

    print(
        "\niteration:",
        check.get("iter"),
    )

    print(
        "actor params:",
        len(check["actor_state_dict"]),
    )

    print(
        "critic params:",
        len(check["critic_state_dict"]),
    )

    print("\nDONE.")
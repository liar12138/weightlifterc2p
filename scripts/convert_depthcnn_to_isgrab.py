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
    "2026-08-22_14-10-50/model_convert222.pt"
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

    print("\n======================================")
    print("COPY OLD CNN -> CURRENT CNN")
    print("======================================")

    # ======================================================
    # 1. 只复制 CNN
    #
    # old:
    #   cnn.xxx
    #
    # current:
    #   cnn.xxx
    # ======================================================

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

    print("\n======================================")
    print("COPY OLD MLP -> CURRENT MLP_TRACKING")
    print("======================================")

    # ======================================================
    # 2. old mlp -> current mlp_tracking
    #
    # old:
    #   mlp.0.weight
    #   mlp.0.bias
    #   ...
    #
    # current:
    #   mlp_tracking.0.weight
    #   mlp_tracking.0.bias
    #   ...
    # ======================================================

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

    # 写回 actor
    current_ckpt["actor_state_dict"] = current_actor

    # ======================================================
    # 3. 其他东西全部保留 model_1100.pt
    # ======================================================

    print("\n======================================")
    print("KEEP FROM CURRENT model_1100.pt")
    print("======================================")

    print("[KEEP] fk_cnn")
    print("[KEEP] mlp_grap")
    print("[KEEP] obs_normalizer")
    print("[KEEP] distribution")
    print("[KEEP] critic")
    print("[KEEP] optimizer")
    print("[KEEP] iter")

    # ======================================================
    # 4. 验证 CNN
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

        print(
            f"{old_key:40s}: {same}"
        )

        if not same:
            cnn_ok = False

    # ======================================================
    # 5. 验证 MLP tracking
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

    if not cnn_ok:
        raise RuntimeError("CNN 权重验证失败")

    if not tracking_ok:
        raise RuntimeError("mlp_tracking 权重验证失败")

    # ======================================================
    # 6. 保存
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
    # 7. 重新加载验证
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
import torch


OLD_MODEL = (
    "/data/weightlifter/logs/g1_tracking/rsl_rl/"
    "2026-08-21_16-31-23-H2-Camera/model_32000.pt"
)

NEW_MODEL = (
    "/data/weightlifter/logs/g1_tracking/rsl_rl/"
    "2026-08-22_00-02-43/model_0.pt"
)

SAVE_MODEL = (
    "/data/weightlifter/logs/g1_tracking/rsl_rl/"
    "2026-08-22_01-39-25/model_convert.pt"
)


def transfer_actor_weights(old_actor, new_actor):
    """
    old:
        obs_normalizer
        distribution
        cnn
        mlp

    new:
        fk_cnn
        obs_normalizer
        distribution
        cnn
        mlp_tracking
        mlp_grap

    映射：
        old obs_normalizer -> new obs_normalizer
        old distribution   -> new distribution
        old cnn            -> new cnn
        old mlp            -> new mlp_tracking
        old mlp            -> new mlp_grap

    fk_cnn:
        保留新模型 model_0.pt 自己的初始化
    """

    result = {}

    print("\n========== TRANSFER ACTOR ==========")

    for new_key, new_value in new_actor.items():

        # ==========================================
        # 1. fk_cnn 不从旧模型复制
        # ==========================================
        if new_key == "fk_cnn":
            result[new_key] = new_value.clone()

            print(
                f"[KEEP NEW] {new_key} "
                f"{tuple(new_value.shape)}"
            )

            continue

        # ==========================================
        # 2. mlp_tracking / mlp_grap
        #    都从 old mlp 复制
        # ==========================================
        if new_key.startswith("mlp_tracking."):

            suffix = new_key[len("mlp_tracking."):]
            old_key = "mlp." + suffix

        elif new_key.startswith("mlp_grap."):

            suffix = new_key[len("mlp_grap."):]
            old_key = "mlp." + suffix

        # ==========================================
        # 3. CNN / normalizer / distribution
        #    key 不变
        # ==========================================
        else:
            old_key = new_key

        # ==========================================
        # 检查 key
        # ==========================================
        if old_key not in old_actor:
            raise KeyError(
                f"\n旧 actor 中不存在:\n"
                f"    {old_key}\n"
                f"无法复制到:\n"
                f"    {new_key}"
            )

        old_value = old_actor[old_key]

        # ==========================================
        # 检查 shape
        # ==========================================
        if old_value.shape != new_value.shape:
            raise ValueError(
                f"\nShape 不匹配:\n"
                f"old: {old_key} "
                f"{tuple(old_value.shape)}\n"
                f"new: {new_key} "
                f"{tuple(new_value.shape)}"
            )

        result[new_key] = old_value.clone()

        print(
            f"[LOAD] {old_key:35s}"
            f" -> {new_key:40s}"
            f" {tuple(old_value.shape)}"
        )

    return result


def compare_actor(old_actor, new_actor):
    """
    转换完成后检查关键参数是否完全相等。
    """

    print("\n========== VERIFY ==========")

    checks = [
        (
            "obs normalizer mean",
            "obs_normalizer._mean",
            "obs_normalizer._mean",
        ),
        (
            "obs normalizer std",
            "obs_normalizer._std",
            "obs_normalizer._std",
        ),
        (
            "cnn conv_h",
            "cnn.conv_h.weight",
            "cnn.conv_h.weight",
        ),
        (
            "cnn conv_v",
            "cnn.conv_v.weight",
            "cnn.conv_v.weight",
        ),
        (
            "cnn fc",
            "cnn.fc.2.weight",
            "cnn.fc.2.weight",
        ),
        (
            "tracking mlp first layer",
            "mlp.0.weight",
            "mlp_tracking.0.weight",
        ),
        (
            "grab mlp first layer",
            "mlp.0.weight",
            "mlp_grap.0.weight",
        ),
        (
            "tracking mlp output",
            "mlp.6.weight",
            "mlp_tracking.6.weight",
        ),
        (
            "grab mlp output",
            "mlp.6.weight",
            "mlp_grap.6.weight",
        ),
        (
            "distribution",
            "distribution.std_param",
            "distribution.std_param",
        ),
    ]

    all_ok = True

    for name, old_key, new_key in checks:

        same = torch.equal(
            old_actor[old_key],
            new_actor[new_key],
        )

        print(
            f"{name:30s}: {same}"
        )

        if not same:
            all_ok = False

    return all_ok


if __name__ == "__main__":

    # ======================================================
    # 1. 加载两个 checkpoint
    # ======================================================

    print("Loading old model:")
    print(OLD_MODEL)

    old_ckpt = torch.load(
        OLD_MODEL,
        map_location="cpu",
    )

    print("\nLoading new model template:")
    print(NEW_MODEL)

    new_ckpt = torch.load(
        NEW_MODEL,
        map_location="cpu",
    )


    # ======================================================
    # 2. 打印 checkpoint 外层
    # ======================================================

    print("\n========== OLD CHECKPOINT ==========")
    print(old_ckpt.keys())

    print("\n========== NEW CHECKPOINT ==========")
    print(new_ckpt.keys())


    # ======================================================
    # 3. actor 权重迁移
    # ======================================================

    old_actor = old_ckpt["actor_state_dict"]
    new_actor = new_ckpt["actor_state_dict"]

    transferred_actor = transfer_actor_weights(
        old_actor,
        new_actor,
    )

    new_ckpt["actor_state_dict"] = transferred_actor


    # ======================================================
    # 4. critic
    #
    # critic 网络结构没变，所以复制旧 critic。
    # ======================================================

    old_critic = old_ckpt["critic_state_dict"]
    new_critic = new_ckpt["critic_state_dict"]

    # 先检查结构
    critic_ok = True

    for key, value in old_critic.items():

        if key not in new_critic:
            print(
                f"[WARNING] new critic 缺少 {key}"
            )
            critic_ok = False
            break

        if value.shape != new_critic[key].shape:
            print(
                f"[WARNING] critic shape mismatch:"
                f" {key}"
            )
            critic_ok = False
            break

    if critic_ok:

        print("\n[INFO] Copy old critic.")

        new_ckpt["critic_state_dict"] = {
            key: value.clone()
            for key, value in old_critic.items()
        }

    else:

        print(
            "\n[WARNING] Critic 结构不同，"
            "保留 model_0.pt 的 critic。"
        )


    # ======================================================
    # 5. optimizer
    #
    # !!!!! 非常重要 !!!!!
    #
    # 不要复制 OLD optimizer。
    #
    # 当前 new_ckpt 本身来自新网络 model_0.pt，
    # 所以它自己的 optimizer_state_dict 与：
    #
    #   cnn
    #   fk_cnn
    #   mlp_tracking
    #   mlp_grap
    #
    # 的参数数量是匹配的。
    #
    # 因此这里什么都不要做。
    # ======================================================

    print(
        "\n[INFO] Keep NEW model optimizer_state_dict."
    )


    # ======================================================
    # 6. iteration
    #
    # 因为这是迁移初始化，不是真正从 32000 iteration
    # 原地续训，所以建议从 0 开始。
    # ======================================================

    #new_ckpt["iter"] = 0


    # ======================================================
    # 7. 验证 actor
    # ======================================================

    verify_ok = compare_actor(
        old_actor,
        new_ckpt["actor_state_dict"],
    )


    if not verify_ok:
        raise RuntimeError(
            "权重验证失败，不保存 checkpoint！"
        )


    # ======================================================
    # 8. 保存
    # ======================================================

    torch.save(
        new_ckpt,
        SAVE_MODEL,
    )


    print("\n====================================")
    print("转换成功")
    print("====================================")

    print(
        f"\nSaved to:\n{SAVE_MODEL}"
    )


    # ======================================================
    # 9. 再重新 load 一遍确认文件没问题
    # ======================================================

    check = torch.load(
        SAVE_MODEL,
        map_location="cpu",
    )

    print("\nSaved checkpoint keys:")
    print(check.keys())

    print(
        "\noptimizer_state_dict exists:",
        "optimizer_state_dict" in check,
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
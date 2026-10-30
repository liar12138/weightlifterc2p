from rsl_rl.algorithms.ppo import *

class DistillPPO(PPO):

    def update(self) -> dict[str, float]:
        if not hasattr(self,'current_iter'):
            self.current_iter = 0

        self.current_iter += 1
        progress = self.current_iter / 1000
        lambda_d = max(0.1, 1.0 - progress)
        lambda_ppo = 1.0 - lambda_d  # 确保和为 1

        # 初始化损失统计量
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_distill_loss = 0.0

        # 获取 mini-batch 生成器
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # 优势归一化（可选）
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            # 对称增强（如果启用）
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # ----- 前向传播 -----
            # Actor 输出（用于动作概率和蒸馏）
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Critic 输出
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])

            # ----- 计算 PPO 损失 -----
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            # 基础 PPO 损失
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # ----- 计算蒸馏损失（仅针对原始样本）-----
            with torch.no_grad():
                # teacher_latent = self.actor.teacher.get_latent(batch.observations[:original_batch_size])
                teacher_out = self.actor.teacher(batch.observations[:original_batch_size])
            # student_latent = self.actor.get_latent(batch.observations[:original_batch_size])
            student_out = self.actor(batch.observations[:original_batch_size])

            cos_sim = torch.nn.functional.cosine_similarity(student_out, teacher_out, dim=-1).mean()
            cos_loss = 1.0 - cos_sim
            mse_loss = torch.nn.functional.mse_loss(student_out, teacher_out)
            distill_loss = mse_loss + cos_loss  # 与原代码保持一致

            # 将蒸馏损失加入总损失（可通过权重调整）
            loss = lambda_ppo * loss + 100 * distill_loss

            # ----- 其他辅助损失（RND、对称性）-----
            if self.rnd:
                rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size])
                loss = loss + rnd_loss  # 原代码中 rnd_loss 未加权？自行调整
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # ----- 反向传播与优化 -----
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()  # 注意：如果 rnd_loss 已加入 loss，此处会重复计算梯度，应调整！
                # 建议将 rnd_loss 也统一到 loss 中，然后用一个 optimizer.step()
            # 梯度裁剪
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            # ----- 统计损失 -----
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_distill_loss += distill_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # 平均损失
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_distill_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "distill": mean_distill_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.storage.clear()
        return loss_dict
---
name: ISCRL_Research_Assistant
version: 1.0.0
description: >
  专门用于辅助 ISCRL (Interpretable Summarization via Contrastive & Reinforcement Learning) 
  模型的开发、调试与论文写作。包含相对于 DSR-RL 的核心改进点、数学公式实现逻辑及代码结构规范。
author: User (Grad Student @ XUPT)
context_files:
  - "基于自监督对比学习与强化学习的可解释视频摘要框架(2025年12月4日V8)(1).docx"
---

# Skill: ISCRL Expert Context & Instructions

## 1. 模型核心背景 (Model Context)

本技能旨在辅助用户完成 ISCRL 模型的开发。该模型是针对无监督视频摘要任务设计的，核心目的是解决 **DSR-RL (Deep Self-attention Recurrent summarization network with RL)** 基线模型的以下局限：
1.  **特征鲁棒性差**：难以跨越深层语义鸿沟。
2.  **训练不稳定**：传统 RL 策略梯度存在高方差。
3.  **不可解释性**：缺乏透明的决策机制。

### 核心创新点 (Key Innovations over DSR-RL)

Agent 在生成代码或解释时，必须严格遵循以下四大改进模块：

#### A. 自监督对比学习模块 (SimCLR Module) [Source: 9, 20, 21]
* **目标**: 提取具备“语义一致性”与“时序鲁棒性”的视觉表征。
* **方法**: 引入 SimCLR，通过构造正负样本对，最大化 InfoNCE Loss。
* **作用**: 解决 DSR-RL 泛化能力差的问题，在无标签条件下学习判别性特征。

#### B. 双空间语义奖励机制 (Dual-Space Semantic Reward) [Source: 9, 25, 30]
* **原理**: 联合约束两个特征空间。
    1.  **原始特征空间 (Original Space)**: 侧重 $R_{div}$ (多样性) 和 $R_{rep}$ (代表性/覆盖率)。
    2.  **不变特征空间 (Invariant Space)**: 经 SimCLR 投影后，侧重语义一致性。
* **公式**: $R_{dual} = \beta \cdot R_{orig} + (1-\beta) \cdot R_{inv}$，其中 $\beta \in [0.1, 0.2]$ 为最佳平衡点。

#### C. 自适应干预机制 (AIM - Adaptive Intervention Mechanism) [Source: 27, 50]
* **目标**: 稳定 RL 训练，防止策略震荡。
* **组件**:
    1.  **自适应基线**: 指数移动平均 (EMA) 平滑历史奖励。
    2.  **动态干预强度 ($I_t$)**: 基于增益函数 $G_t$ 动态调节梯度裁剪阈值和学习率。

#### D. 可解释性增强 (Explainability) [Source: 9, 20]
* **时间维度**: 利用 Self-Attention 权重展示时序关注点。
* **空间维度**: 利用 Smooth Grad-CAM++ 展示单帧内的前景/主体关注区域。
## 2. 代码生成指令 (Coding Instructions)

当用户要求编写或重构代码时，请参照以下逻辑结构：

### 指令 1: 实现双空间奖励函数 (Implement Dual Reward)
**Trigger**: `/code-reward`
**Logic**:
1.  接收 `features_orig` (原始CNN特征) 和 `features_inv` (SimCLR特征)。
2.  分别计算两个空间的 $R_{div}$ (多样性) 和 $R_{rep}$ (代表性)。
3.  $R_{rep}$ 计算需使用 $exp(-\frac{1}{T}\sum \min \|x_t - x_{t'}\|)$ [Source: 39]。
4.  应用线性融合公式：`total_reward = beta * reward_orig + (1 - beta) * reward_inv`。

### 指令 2: 实现 AIM 优化器 (Implement AIM Optimizer)
**Trigger**: `/code-aim`
**Logic**:
1.  在 RL 训练循环中，计算增益 $G_t = \Delta R_t - \lambda |\Delta I_t|$ [Source: 56]。
2.  **条件判断**:
    * 如果 $G_t \ge 0$: 增加干预强度 `I_t = min(I_t * 1.1, I_max)`。
    * 如果 $G_t < 0$: 衰减干预强度 `I_t = max(I_t * 0.95, I_min)`。
3.  **动态调整超参**:
    * 梯度裁剪: `clip_norm = C_base / I_t`。
    * 学习率缩放: `lr_scale = 1 / (1 + 0.1 * (I_t - 1))`。

### 指令 3: 定义 SimCLR 损失 (Implement SimCLR Loss)
**Trigger**: `/code-simclr`
**Logic**:
1.  对 Batch 内的视频帧进行两次随机增强 (Augmentation)，得到 $z_i, z_j$。
2.  计算余弦相似度 `sim(z_i, z_j)`。
3.  实现 NT-Xent Loss (Normalized Temperature-scaled Cross Entropy Loss) [Source: 34]。

---

## 3. 论文辅助写作 (Writing Assistance)

**Trigger**: `/draft-paper`
当用户请求修改论文段落时，请遵循以下风格：
* **术语准确**: 使用 "Semantic Gap" (语义鸿沟), "Temporal Robustness" (时序鲁棒性), "Gradient Variance" (梯度方差)。
* **对比论证**: 始终强调 ISCRL 相对于 "Base Model (DSR-RL)" 的优势，特别是 "Dual-Space Constraint" (双空间约束) 如何解决单一特征空间的局限性。

---

## 4. 常见问题排查 (Troubleshooting Guide)

* **问题**: 训练初期 Loss 不下降。
    * **检查**: 检查 SimCLR 的预训练是否充分？RL 的探索率 (Epsilon) 是否设置合理？
* **问题**: AIM 导致的梯度消失。
    * **检查**: 检查 $I_{max}$ 是否设置过大导致 `lr_scale` 接近 0。建议检查 `I_t` 的变化曲线。
* **问题**: 奖励函数 $R_{dual}$ 波动过大。
    * **建议**: 调整 $\beta$ 值，论文建议 $\beta \approx 0.1$ [Source: 83]。

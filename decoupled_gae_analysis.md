# 大模型强化学习环境交互中的“双轨/解耦 GAE”架构解析

在大模型与外部环境交互（如 Search-R1 的检索反馈、ReAct 框架等）的强化学习训练中，处理长文本外部反馈（Info/Observation）是一个尤为棘手的问题。本篇文档系统性总结了关于 Skip-Info GAE 的难点、理论误区，以及理想的解决方案：双轨/解耦 GAE（Double-track GAE）。

---

## 1. 核心困境：长文本 Info 对 RL 信号的破坏

在典型的 Actor-Critic (PPO) 框架下，模型从环境中获得反馈（数百至数千 Tokens 的搜索结果 `Info`）。这带来了两个关键冲突：

1. **Actor（策略网络）不该为 Info 负责**：Info 是环境生成的，并非 Actor 的动作。如果不加筛选地对 Info 计算 Policy Loss，等同于强迫模型学习“背诵”搜索结果，完全摧毁了策略优化。
2. **长文本造成的 Advantage 衰减**：在标准 GAE 算法中，由于 $\gamma \lambda$ 的步长衰减效应（如 $0.99 \times 0.95 \approx 0.94$），在文章最末处的最终得分（Reward=1.0），穿越几百个 Info Token 往前传导时，会呈指数级衰减趋近于 0。这导致触发搜索动作的 `<search>` Token 根本拿不到长期的学分（Credit Assignment 失败）。

---

## 2. 直觉误区：为什么粗暴使用 `info_mask` 平替会引发灾难？

针对上述问题，最直觉的改法是将所有的 `eos_mask` 或 `attention_mask` 全部替换为 `info_mask`（在其上 Info 部分标记为 0）。
但这在数学上会彻底摧毁 Critic 的训练，形成以下连锁灾难：

1. **Return 目标崩坏**：
   如果用 `info_mask` 计算 Critic 的 Return 目标，Info 部分的 Advantage 和 $\delta$ 会被强行熔断。
   目标公式退化为：`Return_info = A_future + V_old_info`。这导致 Return 和自身的旧预测重合，脱离了真实的 Reward 预期。
2. **Critic 失去学习能力**：
   配合训练时的 `loss_mask`，Critic 在 Info 这几百个 token 上的 Loss 为 0。它彻底变成了不用学习的随机噪声网络，既无法认识“什么是好的搜索资料”，更无法有效给出评分。
3. **Actor 遭受连带反噬**：
   Actor 计算它自己动作（如 `<search>` 的最后一步）的 $\delta$ 时，公式为：`delta = r + gamma * V_{info_start} - V_{search}`。由于 Critic 没在学，这里的 `V_{info_start}` 全是随机数，彻底毁了 Actor 辛辛苦苦的 Skip-Info 穿透隧道传来的价值。

---

## 3. 终极解决方案：双轨/解耦 GAE（Double-track GAE）

为了同时满足“Actor 跨越 Info 衰减”与“Critic 充分学习 Info 价值”，两者必须在 GAE 计算阶段分道扬镳，采用解耦的两套逻辑：

### 轨道 1：Critic 的视角（使用原生 `attention_mask`）
* **计算逻辑**：使用 `attention_mask` 跑一遍标准 GAE，得到真正的 $\delta$ 和完整的衰减 Advantage_{critic}。
* **Return 目标**：`Returns = Advantage_{critic} + Values`。只有这样，Return 才真正包含了后方真实的累积奖赏。
* **损失计算**：在计算 Critic MSE Loss 时，也用 `attention_mask` 算 Loss。
* **数学意义**：逼迫 Critic 逐词阅读 Info，当它看到文档中出现“关键破案线索”时，它的 Value 会猛涨。Critic 借此学会了**评估环境**。

### 轨道 2：Actor 的视角（使用 `info_mask` 实现虫洞效应）
* **计算逻辑**：使用 `info_mask` 再跑一遍 GAE 计算，特供 Actor 使用。
* **公式精妙处拆解**：
  核心代码为: `lastgaelam = m * (delta + gamma * lam * lastgaelam) + (1 - m) * lastgaelam`
  当遇到 Info 时 ($m=0$)，公式变成了：`lastgaelam = 0 + 1.0 * lastgaelam`。
  这里发生了两个至关重要的魔法：
  1. **$\delta$ 被置 0**：不在 Info 内部计算不属于 Actor 责任的优势。
  2. **衰减倍率变为 1.0**：未来的 Advantage 跨越几百个 Info token 时，不再乘以 $\gamma \lambda$，而是 $100\%$ 无损透传（虫洞效应）！精准奖励到前面的 `<search>` token 上。
* **损失计算**：算 Policy Loss 时使用 `info_mask`，确保梯度不流经环境反馈。

---

## 4. 当前系统逻辑的查漏补缺 (Loopholes Check)

我们在梳理完双轨架构后，二次审视整个流程，看看当前逻辑是否还存在隐患或漏洞：

### 漏洞检查 1：`generation.py` 中 `info_mask` 的张量物理对齐情况
**问题检测**：如果在生成过程中，Info 被单纯抽空，可能会导致序列发生左对齐/长度坍缩，进而使得 `info_mask` 和 Critic 输出的 `Values` 在长度和 token 位置上无法 1对1 匹配。
**结论：安全（已修复/验证）**。根据终端使用 Python 测试 `generation.py` 中 `_info_masked_concatenate_with_padding` 的 argsort 逻辑，代码巧妙使用了基于真实历史的统一 `sorted_indices` 排序。真实轨迹和被 Mask 填平的轨迹在最终输出时维度大小与 Token 物理位置**完全一致（就地挖空）**，不会有错位风险。

### 漏洞检查 2：双轨计算的代码落地重构需求
**问题检测**：目前 `verl/trainer/ppo/core_algos.py` 中 `compute_gae_advantage_return` 是捆绑返回 `adv` 和 `returns` 的。
如果传入 `info_mask`：则返回值 `returns` 会出错。
如果传入 `attention_mask`：则 Actor 无损虫洞会失效。
**行动建议**：
在 Trainer (如 `ray_trainer.py`) 中，应当调用两次（或重构该函数）。
1. `adv_actor, _ = compute_gae_advantage_return(..., mask=info_mask)` (专供 Policy Loss)
2. `_, returns_critic = compute_gae_advantage_return(..., mask=attention_mask)` (专供 VF Loss)
**这是后续代码改造的关键动作。如果没有解耦这个计算步骤，逻辑将永远在“坑 Actor”或“坑 Critic”两者间摇摆。**

### 漏洞检查 3：KL 散度控制的对应域
**问题检测**：Actor 在用 `info_mask` 训练，那么 KL 控制也必须保持步调一致。
**结论：安全**。在 KL 计算时必须且只能在 Actor 的 action token 上生效，因为参考模型 (Reference Model) 同样不该对环境 Info 部分产生惩罚。当前 `verl/trainer/ppo/ray_trainer.py` 中的 `apply_kl_penalty` 应该严格只取 `info_mask`。

---

## 结论
大语言模型加上强化学习环境反馈时，Critic 必须做“入局者”（看懂全局所有字），Actor 必须做“独善其善者”（只对自己说出的话拿奖励算 loss）。
通过拆分掩码构建双轨 GAE，我们能解决“信用无法跨超长 Info 分配”的世纪难题，同时保证价值网络能够准确评估外接搜索源的资料质量。这一洞察对 RAG + RL 的收敛至关重要。



本质上和原来相比就是改变了actor的优势计算，现在actor和critic用的不再是一套优势了。actor的目标函数计算会比原来大，因为穿过了info的衰减

---

## 简历更新建议

**技术栈：** Python, PyTorch, vLLM, VeRL, Ray, PPO/GRPO, Qwen2.5-3B, E5-Retriever

**项目背景：** 针对大语言模型在多跳问答中的推理断层与幻觉问题，本项目基于 veRL 框架在 8x RTX 3090 环境下构建了“推理-检索-再推理”的闭环强化学习流程。通过 PPO 与 GRPO 算法，训练 Qwen2.5-3B 模型在 NQ 与 HotpotQA 任务中实现自主触发检索指令，并动态整合外部反馈进行序贯推理，显著提升了模型对长链条知识的检索增强推理能力。

**核心职责与技术突破：** 基于框架 Rollout 接口实现并优化了自定义交错采样逻辑，实时解析控制 Token 构建模型与外部检索工具的异步交互闭环。在训练工程中，提出并实现了“双轨 GAE (Double-track GAE)”架构以解决长文本环境反馈带来的信用分配难题：通过精细的张量级就地掩码（in-place Mask），使 Critic 网络在全视野下逐词评估检索文本的价值，同时在 Actor 策略优化中重构 GAE 递归逻辑搭建奖励穿透“虫洞”，使优势信号无损跨越数百 Token 的外部环境噪声，精准反馈至早期的搜索决策点。该创新方案有效避免了传统掩码导致的 Critic 退化和奖励耗散问题，经过对比实验，模型在约 280 steps 内实现稳定收敛，测试集 EM 分数由基线的 0.55 跃升至 0.675。
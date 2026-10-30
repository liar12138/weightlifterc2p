# 论文草稿（中文版）：基于宇树 G1 与 H2 的人形机器人举重

> 工作草稿（ICLR 风格）。方括号内为待填字段，须用实测值替换；投稿前请逐一核对参考文献。文献检索截至 2026-08-27。与英文版 `paper_sections_draft.md` 保持同步。

---

## 摘要（Abstract）

人形机器人越来越多地被用来评估其行走、灵巧操作与动态全身技能，但在"运动会式举重"这类场景中的表现，作为具身智能的基准仍未得到充分探索。与普通的物体搬运不同，举重要求双足机器人从地面提起负载，协调双手抓握与快速变化的质心，产生足够的竖直冲量，在接杠与头顶锁定阶段维持平衡，并满足显式的任务与安全约束。我们提出 [METHOD NAME]，一个用于在宇树 G1 与 H2 平台上实现自主人形举重的闭环全身控制框架。该框架将 [运动/参考生成]、[负载感知的状态估计或系统辨识] 与 [物理信息引导的策略/控制自适应] 相结合，以在负载变化下协调运动、姿态、手臂动作与接触力。我们进一步提出 [BENCHMARK NAME]，一个可复现的协议，覆盖负载质量、杠铃高度、握杠配置、举法、头顶稳定与失败恢复等维度。在仿真与真实机器人上的实验表明，我们的方法在 [N] 次试验中达到 [SUCCESS RATE] 成功率，将最大可稳定负载提升 [X]%，并在 [跟踪误差 / 峰值压力中心偏移 / 恢复时间] 上优于 [基线方法]。消融实验表明，[关键组件] 对于在负载引起的动力学偏移下保持鲁棒性至关重要。我们的结果将运动会式举重定位为一个紧凑但极具挑战性的基准，用于研究人形机器人的动态移动操作、负载自适应与安全全身控制。

---

## 1. 引言（Introduction）

人形机器人被设计用来在为人而建的空间中工作，而有用行为远不止孤立的运动或固定基座操作。强化学习、基于模型的控制、运动模仿与仿真到现实的迁移等领域的最新进展，使人形机器人日益敏捷：它们可以行走、从扰动中恢复、操作物体并复现富有表现力的人类动作。然而，大多数评测仍聚焦于自由空间运动、轻量物体交互或准静态操作。这些场景无法充分暴露这样一种耦合动力学：双足机器人必须在与地面产生作用力的同时，加速并稳定一个可观质量的负载。

运动会式举重恰恰是检验这一能力的绝佳测试。任务并非简单地把物体举离地面。一次成功的试举要求机器人选择合适的握距与站距，下蹲接近杠铃，建立双侧接触，同时加速杠铃与自身身体，在负载改变系统惯量的情况下调节全身动量，并将杠铃保持到规定的头顶位置。因此，机器人必须解决一个混合、接触丰富的综合问题，涉及运动、双手操作、平衡、执行器极限与任务相关运动约束。在时机、双手对称性、杠铃轨迹或质心调节上的微小失误，都可能把一次名义上可行的试举变成跌倒、脱手或危险释放。

这一挑战正当时：宇树 G1、H2 等价格可负担、力控型人形平台使可重复的全身实验更加容易。与此同时，它们紧凑的执行器、有限的关节力矩裕度、受限的末端执行器以及平台特有的动力学，使得从人类运动或无负载人形策略直接迁移并不可靠。一旦抓住杠铃，跟踪无负载深蹲或头顶姿态的控制器就可能失效：负载会移动系统总质心、增大关节与接触力、改变有效惯量，并收窄可恢复状态区域。因此，举重是检验"学习到的控制器是否获得了负载感知的协调能力，而非仅仅记住了运动学轨迹"的有效物理压力测试。

已有工作为这一问题奠定了重要基础。早期人形研究展示了动态举升与整身接触（包括 30 kg 箱体），后来的工作处理了未知物理性质箱体的准静态可举性推理。基于模型的人形控制已发展出动量调节、多接触平衡、分层二次规划、无源性与模型预测全身控制。近来，学习方法把通用运动与移动操作策略迁移到真实人形机器人，也有专门工作通过力觉遥操作与负载感知自适应探索了重物举升。然而，这些努力尚未在现代通用人形平台上，针对自由站立双足机器人的自主运动会式杠铃举升提供统一评测。

在本文中，我们将人形举重作为独立的具身控制问题加以研究，并做出四点贡献：

1. **任务形式化。** 我们将运动会式举重形式化为一系列接触与平衡阶段，并给出显式的成功、安全与头顶保持判据，而非单一"末端高度"目标。
2. **负载感知的全身控制。** 我们提出 [METHOD NAME]，将 [负载估计 / 隐动力学自适应] 与 [全身策略或优化] 相结合，在负载变化时维持可行的抓握、动量、接触与执行器行为。
3. **跨平台评测。** 我们在宇树 G1 与 H2 上评测同一任务定义，将方法改进与平台（embodiment）效应区分开，并同时报告仿真与硬件结果。
4. **基准与分析。** 我们提供一套协议与消融，涵盖负载质量、杠铃轨迹、站距、举升速度、观测可得性与恢复行为，便于未来在具有物理意义的任务上比较人形系统。

我们的核心假设是：稳健举重要求控制器调节"机器人—负载"的耦合动力学，而不仅仅是跟踪类人的关节参考。我们通过无负载与有负载策略、运动学跟踪与负载感知控制、纯仿真与硬件适配模型之间的受控对比来检验该假设。除任务本身外，本研究还追问一个更宏大的问题：现代人形机器人能否在可测的安全包络内，自主执行一项大力、接触丰富的技能？答案将为把人类形态机器人作为通用物理智能体来评测提供具体一步。

---

## 2. 相关工作（Related Work）

### 2.1 人形举升与整身物体交互

人形举升的历史早于近期的学习式人形文献，但重点通常在于重物操作、可举性推理或遥操作，而非自主运动式举重。Ohmura 等人证明了人形机器人可通过整身接触与触觉反馈举起 30 kg 箱体，展示了超越仅末端执行器操作的分布式接触价值 [Ohmura2007]。Arisumi 等人研究了 HRP-2 上通过整身运动进行动态举升，将向上的物体加速度与身体运动结合以减少手臂负载 [Arisumi2008]。这些研究在概念上是与我们最接近的前驱：它们都把负载与人形视为一个耦合动力系统。但它们使用早期平台与任务特化、基于模型的策略，未涉及现代学习式迁移或运动会式杠铃协议。

后续工作研究了举升的细分环节。Kanoulas 等人利用几何与力/力矩信息调整抓握位姿，使抓握点靠近物体质心，从而降低腕部力臂、提升举升可靠性 [Kanoulas2018]。Han 等人处理了相反的问题——人形机器人是否应当举起某物体：通过物理交互估计未知物体属性，并查询仿真生成的可行轨迹 [Han2020]。这些工作为我们的形式化提供了可举性推理与负载估计的动机，但都聚焦于箱体与准静态或单臂操作。相比之下，我们的任务要求双侧持杠、大竖向力运动、自由站立双足平衡与头顶稳定阶段。

近来也有学习式整身操作与举升工作。Plan-Guided Reinforcement Learning 用运动计划引导策略在 Punyo 上翻转并举起大箱体 [Zhang2023]。面向人形箱体移动操作的仿真到现实学习研究了跨运动与接触变化的搬运操作 [Dao2023]。AdaptManip 以在线循环状态估计实现自适应整身物体举升与递送 [Byrd2026]，Sumo 则利用测试时引导研究大而重物体的动态、可泛化整身移动操作 [Zhang2026]。这些工作朝自主负载交互迈进，但在物体几何、双侧对称性、力的方向，以及"原地产生并稳定高竖直负载"的要求上，都与杠铃举重不同。

### 2.2 全身平衡与多接触控制

我们问题的控制基础来自人形全身与多接触控制。Resolved momentum control 通过调节线动量与角动量生成协调的人形运动 [Kajita2004]。Sentis 与 Khatib 提出了面向人类环境中人形机器人的、带优先级操作任务与物理约束的全身控制 [Sentis2006]，后续工作发展了多接触与质心行为的柔顺控制 [Sentis2010]。分层二次规划（Hierarchical Quadratic Programming）为在线人形运动生成提供了计算高效、可处理等式与不等式约束的求解方式 [Escande2014]。OpenSoT 进一步将任务优先级与关节阻抗工具封装为可复用的全身控制库 [Rocchi2015]。

针对接触不确定下的平衡问题，已有无源性全身平衡 [Henze2016]、分层平衡与力/力矩控制 [Lee2015]，以及面向力控双足机器人的模型预测全身控制 [Dantec2022]。这些方法为分配接触力、调节质心、满足摩擦与执行器约束提供了原理性机制，天然适合作为举重的基线与组件：负载会引入外部力，改变动量与支撑约束。我们的不同之处在于：把变化的负载动力学作为任务状态的一部分，并在标准化的高负载试举（而非仅在行走、下蹲或通用多接触运动）中评测最终控制器。

### 2.3 学习、运动模仿与仿真到现实的人形控制

学习式人形控制近年来极大地扩展了真实机器人的动作库。ExBody 从人类运动数据学习富有表现力的全身控制，同时放松腿部跟踪以保持运动与平衡 [Cheng2024]。HumanPlus 用仿真训练的低层控制与人类影子（shadowing）采集数据，供后续人形技能学习使用 [Fu2024]。HOVER 将多种控制模式整合为通用的神经全身控制器 [He2025]，AMO 则在宇树 G1 上把仿真到现实的强化学习与轨迹优化相结合，实现自适应全身控制 [Li2025]。这些结果表明，运动先验与大规仿真模可以减少训练敏捷人形的工程成本。

然而，仅靠运动学模仿并不能保证力的可行性。人类与机器人在形态上存在差异，人类参考通常不编码目标机器人的执行器极限、接触力分配或负载惯量。这一点对举重尤其重要：杠铃并非被动的视觉目标，而是通过双手与机器人耦合的额外动力学物体。因此，我们仅在合适处把运动参考作为先验，并评测控制器在负载变化与平台切换下的稳定性。

### 2.4 负载自适应、物理交互与安全

负载变化是人形部署中特别重要的一类动力学偏移。关于人—机人因工程学负载举升的工作，建模了机器人硬件、人体几何与负载位置之间的交互 [Sartore2022; Rapetti2023]。全身双侧遥操作与力觉遥操作也已被用于让轮式人形举起杠铃与箱体，为操作者提供负载引起的平衡线索 [Purushottam2025]。这些系统说明杠铃举升对人形研究具有物理意义，但其控制权威仍主要来自人类引导，且轮式基座回避了双足完全欠驱动的平衡问题。

在控制层面，导纳与阻抗方法显式调节交互力，并在不确定接触下改善柔顺性 [Keemink2018]。更新的工作分析了物体质量对人形平衡的影响，并将平衡状态边界嵌入轨迹优化，用于"举起—保持"与"举起—释放"任务 [Song2026]。这与我们高度相关，应被视为最接近的当代分析性研究。我们的贡献与之互补：不是只关注质量相关的平衡边界，而是在现代双足平台上研究自主的学习式或混合控制器，并用运动会式举重暴露负载自适应、动态运动生成与真实迁移之间的相互作用。

### 2.5 本文定位

综上，文献中已存在五类强先例：(i) 动态整身举升；(ii) 抓握与可举性推理；(iii) 多接触平衡；(iv) 学习式人形运动；(v) 负载感知交互。空白在于：在可复现、IWF 启发的协议下，对自主双足杠铃举升进行联合评测。据我们所知，这是首个同行评议的、在全尺寸双足人形（宇树 H2）上、基于 IWF 启发式协议进行自主杠铃举升的研究，G1 作为额外的小型平台案例。这里有两点需说明。其一，FIRA HuroCup 是以奥运会项目为人形机器人任务来源的联赛，包含面向小型人形的举重项目 [Christmann2022]；因此我们的宣称限定于全尺寸平台与显式形式化的举重协议，而非泛指"人形举重"。其二，厂商发布的力度与负载搬运演示视频确实存在，但它们并非同行评议、符合规则的学术研究。我们保留"据我们所知（to our knowledge）"这一限定词，并建议投稿前对 IEEE Xplore、ACM Digital Library、Scopus/Web of Science、Google Scholar、arXiv 与中文数据库做最终检索。

---

## 3. 待补充章节（修改路线提醒，非正文）

> 按评审意见，正文尚需补写以下章节后再回填摘要。此节仅为工作提示，不进入投稿正文。

- **Problem Formulation**：选定举法变体（建议"双脚不离地的 power snatch"或简化推举）、定义相位（setup → pull → transition → overhead catch/lockout → stabilize → rack）、成功/失败判据、负载递增协议。
- **可行性分析**：由关节力矩上限与 CoP/ZMP 约束反推最大可举质量（G1 单臂额定 2–3 kg；H2 臂额定约 7 kg / 峰值约 15 kg——两手各持约 7.5 kg 的 15 kg 女杆接近额定，20 kg 男杆偏极限；动态上拉段动载荷约为静载 1.5–2 倍）。若使用轻定制杆，则宣称应写为"IWF 启发（inspired）"，不可写"符合人类规则"。
- **Method**：观测（本体感知+关节力矩/电流+IMU，可选深度相机）→ 负载表征（基于动量动力学的杠铃质量/外力估计，或训练期随机负载注入学习负载隐变量）→ 相位与负载条件化的整身 RL 策略（动作空间=关节位置/PD 目标或关节力矩；质量课程 + 域随机化）→ 安全滤波（ZMP/CoP 约束或 CBF）→ 头顶保持阶段按力矩裕度降速/放弃决策。
- **Sim-to-Real**：Isaac Gym/Lab + 系统辨识 + 可选 few-shot 动力学对齐或残差微调。
- **Experiments**：成功率、最大可稳定负载（体重百分比）、头顶保持时间、杠铃轨迹偏差、峰值 CoP 偏移、关节力矩利用率；基线与消融（load-blind vs load-aware、有无相位条件、运动学跟踪 vs 动态控制、sim-only vs sim2real、G1 vs H2）。

---

## 4. 引用清单（与英文版一致，文献本身为英文）

### 基础引用（正文已引用）
- [Ohmura2007] Yoshiyuki Ohmura, Akihiko Nagakubo, Yasuo Kuniyoshi. *Humanoid robot which can lift a 30kg box by whole body contact and tactile feedback*. IROS, 2007. DOI: 10.1109/IROS.2007.4399592.
- [Arisumi2008] Hitoshi Arisumi, Sylvain Miossec, Jean-Rémy Chardonnet, Kazuhito Yokoi. *Dynamic lifting by whole body motion of humanoid robots*. IROS, 2008. DOI: 10.1109/IROS.2008.4651195.
- [Kajita2004] Shuuji Kajita et al. *Resolved momentum control: humanoid motion planning based on the linear and angular momentum*. IROS, 2003/2004. DOI: 10.1109/IROS.2003.1248880.
- [Sentis2006] Luis Sentis, Oussama Khatib. *A whole-body control framework for humanoids operating in human environments*. ICRA, 2006. DOI: 10.1109/ROBOT.2006.1642100.
- [Sentis2010] Luis Sentis, Jaeheung Park, Oussama Khatib. *Compliant control of multicontact and center-of-mass behaviors in humanoid robots*. IEEE T-Robotics, 2010. DOI: 10.1109/TRO.2010.2043757.
- [Escande2014] Adrien Escande, Nicolas Mansard, Pierre-Brice Wieber. *Hierarchical quadratic programming: Fast online humanoid-robot motion generation*. IJRR, 2014. DOI: 10.1177/0278364914521306.
- [Lee2015] Yisoo Lee, Soonwook Hwang, Jaeheung Park. *Balancing of humanoid robot using contact force/moment control by task-oriented whole body control framework*. Autonomous Robots, 2015. DOI: 10.1007/s10514-015-9509-1.
- [Rocchi2015] Alessio Rocchi et al. *OpenSoT: A whole-body control library for the compliant humanoid robot COMAN*. ICRA, 2015. DOI: 10.1109/ICRA.2015.7140076.
- [Henze2016] Bernd Henze, Máximo A. Roa, Christian Ott. *Passivity-based whole-body balancing for torque-controlled humanoid robots in multi-contact scenarios*. IJRR, 2016. DOI: 10.1177/0278364916653815.
- [Dantec2022] Ewen Dantec et al. *Whole-Body Model Predictive Control for Biped Locomotion on a Torque-Controlled Humanoid Robot*. Humanoids, 2022. DOI: 10.1109/HUMANOIDS53995.2022.10000129.
- [Kanoulas2018] Dimitrios Kanoulas et al. *Center-of-Mass-Based Grasp Pose Adaptation Using 3D Range and Force/Torque Sensing*. IJHR, 2018. DOI: 10.1142/S0219843618500135.
- [Han2020] Yuanfeng Han, Ruixin Li, Gregory S. Chirikjian. *Can I lift it? Humanoid robot reasoning about the feasibility of lifting a heavy box with unknown physical properties*. IROS, 2020. DOI: 10.1109/IROS45743.2020.9340729.
- [Sartore2022] Carlotta Sartore, Lorenzo Rapetti, Daniele Pucci. *Optimization of Humanoid Robot Designs for Human-Robot Ergonomic Payload Lifting*. Humanoids, 2022. arXiv:2211.13503.
- [Rapetti2023] Lorenzo Rapetti et al. *A Control Approach for Human-Robot Ergonomic Payload Lifting*. ICRA, 2023. DOI: 10.1109/ICRA48891.2023.10161454.
- [Zhang2023] Mengchao Zhang, Jose Barreiros, Aykut Ozgun Onol. *Plan-Guided Reinforcement Learning for Whole-Body Manipulation*. arXiv:2310.12263.
- [Dao2023] Jeremy Dao, Helei Duan, Alan Fern. *Sim-to-Real Learning for Humanoid Box Loco-Manipulation*. arXiv:2310.03191.
- [Fu2024] Zipeng Fu et al. *HumanPlus: Humanoid Shadowing and Imitation from Humans*. arXiv:2406.10454.
- [Cheng2024] Xuxin Cheng et al. *Expressive Whole-Body Control for Humanoid Robots*. RSS, 2024. DOI: 10.15607/RSS.2024.XX.107.
- [He2025] Tairan He et al. *HOVER: Versatile Neural Whole-Body Controller for Humanoid Robots*. ICRA, 2025. arXiv:2410.21229.
- [Li2025] Jialong Li et al. *AMO: Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control*. arXiv:2505.03738.
- [Keemink2018] Arvid Q. L. Keemink, Herman van der Kooij, Arno H. A. Stienen. *Admittance control for physical human-robot interaction*. IJRR, 2018. DOI: 10.1177/0278364918768950.
- [Purushottam2025] Amartya Purushottam et al. *Heavy lifting tasks via haptic teleoperation of a wheeled humanoid*. arXiv:2505.19530.
- [Song2026] Hyunjong Song, William Z. Peng, Joo H. Kim. *Balancing of Humanoid with Object Mass: Trade-off Analyses and Lifting Control*. arXiv:2607.29625.
- [Byrd2026] Morgan Byrd et al. *AdaptManip: Learning Adaptive Whole-Body Object Lifting and Delivery with Online Recurrent State Estimation*. arXiv:2602.14363.
- [Zhang2026] John Z. Zhang et al. *Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation*. arXiv:2604.08508.

### 补充引用（二次检索新增，2026-08-27）

#### 人形机器人体育/竞技技能
- [Christmann2022] Guilherme Christmann, Lin Yu-Ren, Rodrigo da Silva Guerra, Jacky Baltes. *Can a Robot Shoot an Olympic Recurve Bow? A preliminary study*. arXiv:2212.11071.（FIRA HuroCup：以奥运项目为人形任务来源，含小型人形举重项目。）
- [Liu2025Badminton] Chenhao Liu et al. *Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning*. arXiv:2511.11218.
- [Chen2026Badminton] Yeke Chen et al. *Learning Human-Like Badminton Skills for Humanoid Robots*. arXiv:2602.08370.
- [Ren2026CyboRacket] Peng Ren et al. *CyboRacket: A Perception-to-Action Framework for Humanoid Racket Sports*. arXiv:2603.14605.
- [Huang2026Tennis] Tao Huang et al. *Towards Professional Tennis Styles for Humanoid Robots (AdaPT)*. arXiv:2608.20087.
- [Zhong2026RoboNaldo] Yichao Zhong et al. *RoboNaldo: Humanoid Soccer Shooting via Motion-Guided Curriculum RL*. arXiv:2606.11092.
- [Wang2026Marope] Zihao Wang et al. *Cooperative Long Rope Skipping via Multi-Agent RL*. arXiv:2606.08064.
- [Gu2026SKATER] Junchi Gu et al. *SKATER: Roller Skating*. arXiv:2601.04948.

#### 外力/负载估计与动力学自适应
- [Rotella2015] Nicholas Rotella, Alexander Herzog, Stefan Schaal, Ludovic Righetti. *Humanoid Momentum Estimation Using Sensed Contact Wrenches*. HUMANOIDS, 2015. DOI: 10.1109/HUMANOIDS.2015.7363417.
- [Chen2026SixthSense] Xingzhou Chen et al. *SixthSense: Task-Agnostic Proprioception-Only Whole-Body Wrench Estimation for Humanoids*. arXiv:2605.01427.
- [Shi2026Compliance] Haochen Shi et al. *Minimalist Compliance Control*. arXiv:2603.00913.
- [Liu2026LAC] Yang Liu et al. *LAC: Linear and Angular Compliance for Humanoid Whole-body Control*. arXiv:2608.25405.
- [Lyu2026ADAPT] Bofan Lyu et al. *ADAPT: Analytical Disturbance-Aware Policy Training for Humanoid Locomotion*. arXiv:2606.16542.
- [Xie2026FADA] Angchen Xie et al. *FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control*. arXiv:2606.28476.

#### 动态双手操作与平台规格
- [Kim2025ARMADA] Jaehyung Kim et al. *ARMADA: A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation*. arXiv:2502.16908.
- [UnitreeG1] 宇树 G1 规格。https://www.unitree.com/g1/（2026-08-27 访问）。高 1.32 m、约 35 kg、23 DoF（EDU 23–43）、单臂最大负载约 2–3 kg、膝峰扭矩 90/120 N·m。
- [UnitreeH1] 宇树 H1 / H1-2 规格。https://www.unitree.com/h1/（2026-08-27 访问）。H1：约 1.80 m、约 47 kg、最大关节扭矩 360 N·m；H1-2：约 1.78 m、约 70 kg、臂额定约 7 kg / 峰值约 21 kg。
- [UnitreeH2] 宇树 H2 规格。https://www.unitree.com/H2（2026-08-27 访问）。1.82 m、约 70 kg、31 DoF、臂关节扭矩 120 N·m、臂最大负载额定约 7 kg / 峰值约 15 kg。作为"全尺寸"宣称的主平台。

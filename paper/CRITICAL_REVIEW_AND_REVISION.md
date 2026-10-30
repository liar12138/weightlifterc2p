# 论文草稿锐评 + 重新调研 + 修改方案

> 对象：`paper_sections_draft.md`（Abstract / Introduction / Related Work 草稿）
> 日期：2026-08-27。本文档供修改用，最终文字以英文投稿为准。

---

## 一、锐评（按严重程度排序）

### 1.1 优点（先说好的）

1. **定位方向是对的**：把"自主双足运动会式举重"作为独立问题提出，并把它挂在"人形机器人竞技/体能技能（sports skills）"这一正在爆发的趋势上，审稿人容易买账。
2. **引用了经典举重/整身接触文献**（Ohmura2007、Arisumi2008、Kanoulas2018、Han2020、Purushottam2025、Song2026），说明你区分了"历史整身举重"和"现代自主双足举重"，叙述清楚。
3. **有自查意识**：对"第一篇"的说法加了 caveat，这是对的。
4. 结构符合 ICLR 论文习惯（Abstract → Intro 贡献列表 → Related Work 分小节 → 定位）。

### 1.2 P0 级问题（不解决不能投）

**P0-1：论文"没有方法"，本质是 position statement，不是 paper。**
- 全文没有 [METHOD NAME] 的架构、状态/动作空间、奖励/损失、训练流程、控制律、sim-to-real 策略、消融设计。
- ICLR 的核心是"学习/控制方法 + 消融 + 可复现评测"。现在的草稿只是把"我要研究什么"写清了。
- **必须给出具体方法**（见第四节方案），并且摘要里的数字（成功率、最大负载提升、误差降低）必须是真实实验值，不能留空占位。

**P0-2："第一篇"的宣称证据链不足，甚至可能被审稿人直接打脸。**
- 我的检索确认：arXiv 上 `humanoid weightlifting`=0、`"Olympic lifting"+humanoid`=0、`barbell+humanoid`=1（仅轮式遥操作）、`"clean and jerk"`=0、`humanoid snatch`=1（仅一条双臂硬件论文 ARMADA）、`"Unitree H2"`=0。
- **但**：
  1. FIRA HuroCup 联赛就是以"真实奥运项目"为任务的人形联赛（Christmann et al., arXiv:2212.11071 明确写了这一点），其中包含小型人形举重（weightlifting）项目。所以"史上第一个举人形举重"是**假**的。
  2. 宇树官方发布过 H1/G1 举哑铃/杠铃的**演示视频**（非论文，但公开存在，审稿人知道）。
  3. arXiv 查不到 ≠ IEEE Xplore / Scopus / 中文期刊没有；中文文献完全没覆盖。
- **修改方向**：把宣称收窄为"to our knowledge, the first **peer-reviewed** study of **autonomous, competition-style (snatch/clean-and-jerk-like) barbell lifting on full-size torque-controlled bipedal platforms (G1/H2) with a standardized protocol**"，并在正文显式引用 HuroCup 与宇树演示视频以消解"绝对首次"的话术风险。

### 1.3 P1 级问题（决定论文能不能站住）

**P1-1：任务定义太模糊。**
- "运动会举重"在 IWF 里是抓举（snatch）和挺举（clean & jerk）两套完全不同的动作：抓举需要**动态下蹲接杠**、允许脚移动/分腿；挺举含**前蹲→上挺→头顶锁定**，且动作中允许双脚短暂离地。机器人和人都不同。
- **必须选定一个变体**（建议：双脚不可移动的 "power snatch" 或 "strict/overhead press" 简化版，G1 扭矩有限，动态接杠很可能超力矩），并定义：相位序列（setup → pull → transition → overhead catch/lockout → stabilize → rack）、成功判据（杠铃头顶、手臂锁定、双足触地、CoP 在支撑多边形内、保持 X 秒）、失败判据、负载递增协议。

**P1-2：平台可行性完全没有论证（这是最大的技术隐患）。**
- G1 官方参数（已核实）：高 1.32 m，约 35 kg，23 DoF（EDU 23–43），**单臂最大负载约 2–3 kg**，膝峰扭矩 90 N·m（EDU 120 N·m），小腿+大腿 0.6 m。
- 这意味着：直臂头顶锁定重杠铃的**力矩需求远超手臂/肩部能力**；真正的"举重"很可能只能是**很轻的杠铃杆/空杆 + 利用整身动力学的动态借力**，或者干脆是"负重下蹲 + 半程推举"这类变体。
- 审稿人一定会问："你的平台凭什么能举起 X kg？" 需要一节 **actuation & balance feasibility analysis**：由关节力矩上限、支撑多边形、CoP/ZMP 约束反推最大可举质量；可引用 Song2026（质量-平衡边界）、Han2020（可举性推理）、Purushottam2025（21% 体重作为标定参照）。
- 另外 H2 的参数我没有在官网抓到（页面 404），**不要猜**，投稿前用官方规格表核实；G1/H2 形态差异巨大，跨本体论证要小心。

**P1-3：没有评测指标和协议。**
- 需要显式定义：成功率、最大稳定负载、杠铃路径偏差（竖直/横向）、峰值 CoP 偏移/GRF 裕度、关节扭矩利用率、头顶保持时间、扰动恢复成功率、跌倒保护（安全包络）。
- 基线至少应包含：(a) 无负载策略直接迁移（load-blind）；(b) 纯运动学参考跟踪；(c) 不做负载自适应的模型 MPC；(d) 仿真训练 vs sim-to-real；(e) 多 seed、多 trials、硬实验视频。

### 1.4 P2 级问题（提升论文质量）

- **缺少体育类人形机器人相关工作小节**（见第三节），这类工作是你的直接"同趋势对手"，不提会被认为是文献盲区。
- **缺少外力/负载估计相关工作**：举重本质是"负载改变动力学"，Rotella2015 / SixthSense2026 / Minimalist Compliance2026 / LAC2026 / ADAPT2026 / FADA2026 都是必须提的。
- **"G1 和 H2"跨本体声称过强**：若方法没有共享观测/动作空间或明确微调流程，"我们同时做 G1 和 H2"会被当成工作量叙事。要么给出 cross-embodiment 界面，要么改成"主要平台 + 迁移案例"。
- **引用清单不一致**：正文引了 `Dantec2022` 但清单缺失；多篇 arXiv 预印本（Byrd2026、Zhang2026、Song2026、Purushottam2025 等）年份/作者易变，投稿前逐一核对。
- **缺少"为什么难"的技术论证**：应该加一段——负载改变系统 CoM/惯量、收紧 ZMP 约束、逼近力矩饱和，且**双手+杠铃形成闭合运动链**使系统过约束、力分配变得关键（这是举重区别于普通搬运的核心技术点）。

### 1.5 P3 级问题

- 缺图：典型 ICLR 机器人论文至少要有 teaser、方法架构图、结果曲线；草稿完全没规划。
- 缺 ICLR 定位句："我们学到了什么？（a load-conditioned whole-body policy / 负载隐变量自适应）"。纯系统集成论文不适合 ICLR。
- 摘要占位符过多，读完抓不到具体贡献。

---

## 二、重新调研结果

### 2.1 空白确认（2026-08-27 检索）

| 检索词 | arXiv 结果 | 说明 |
|---|---|---|
| `humanoid weightlifting` | 0 | — |
| `"Olympic lifting" AND humanoid` | 0 | — |
| `barbell AND humanoid` | 1 | 仅轮式人形遥操作举重（Purushottam2025） |
| `"clean and jerk"` / `humanoid snatch` | 0 / 1 | 唯一命中的 snatch 是双臂硬件 ARMADA，非人形举重 |
| `humanoid dumbbell` | 0 | — |
| `"Unitree H2"` | 0 | H2 平台无 arXiv 论文 |

→ 结论：**自主双足运动会式杠铃举重这个具体组合，在 arXiv 上确实是空白**；但"人形举重"本身不空白（HuroCup、宇树演示视频、轮式遥操作）。

### 2.2 必须补进 Related Work 的新文献（均已核实）

**A. 人形机器人体育/竞技类（你的直接趋势邻居）**
1. CyboRacket: Perception-to-Action Framework for Humanoid Racket Sports（arXiv:2603.14605，2026，G1 网球）
2. Towards Professional Tennis Styles for Humanoid Robots (AdaPT)（arXiv:2608.20087，2026，G1 网球）
3. Learning Human-Like Badminton Skills for Humanoid Robots（arXiv:2602.08370，2026）
4. Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning（arXiv:2511.11218，2025）——**多阶段课程 RL 训练体育技能，方法论上与你的做法最像，必引**
5. RoboNaldo: Humanoid Soccer Shooting via Motion-Guided Curriculum RL（arXiv:2606.11092，2026，G1）
6. Marope: Cooperative Long Rope Skipping（arXiv:2606.08064，2026，G1）
7. SKATER: Roller Skating（arXiv:2601.04948，2026）
8. Can a Robot Shoot an Olympic Recurve Bow?（arXiv:2212.11071，2022）——**FIRA HuroCup"以奥运项目为任务"的出处，含人形举重项目背景**
9. （可选）TaskNPoint: backhand tennis（arXiv:2606.26215）、HUSKY 滑板（arXiv:2602.03205）

**B. 外力/负载估计与自适应（方法核心依据）**
1. Humanoid Momentum Estimation Using Sensed Contact Wrenches（arXiv:1507.04401，Rotella et al., 2015）
2. SixthSense: Task-Agnostic Proprioception-Only Whole-Body Wrench Estimation（arXiv:2605.01427，2026）
3. Minimalist Compliance Control（arXiv:2603.00913，2026）——无力传感器估外力
4. LAC: Linear and Angular Compliance（arXiv:2608.25405，2026）——外部力/力矩下的整身柔顺
5. ADAPT: Analytical Disturbance-Aware Policy Training（arXiv:2606.16542，2026）
6. FADA: Few-Shot Domain Adaptation via Dynamics Alignment（arXiv:2606.28476，2026）——负载/地形动力学偏移适配

**C. 动态双臂操纵（硬件/操控角度）**
1. ARMADA: low-cost bimanual arm for dynamic & contact-rich manipulation（arXiv:2502.16908，2025）——"snatching"一词与举重直接相关

**D. 平台规格（已核实 G1）**
- G1：1.32 m，~35 kg，23 DoF（EDU 23–43），单臂最大负载 ~2–3 kg，膝峰扭矩 90/120 N·m，小腿+大腿 0.6 m（来源：unitree.com/g1，2026-08-27 抓取）。H2 规格未能抓取，请自行从 unitree.com/H2 核实后引用。

---

## 三、修改意见

### 3.1 方向性选择（先想清楚，再动笔）

**Option A（推荐）——定位为"学习方法 + 基准"：**
> 贡献 = (1) 负载条件化的整身控制方法（负载/外力估计 → 相位与负载条件化的 WBC/RL 策略 → ZMP/CBF 安全滤波）；(2) 标准化的运动会举重协议与指标；(3) G1/H2 仿真+实机评测与消融。

适合 ICLR，审稿人能回答"学到了什么"：**学习把变化的负载动力学纳入整身协调，而不是只跟运动学参考**。

**Option B——定位为"基准/任务 + 系统"：**
> 如果你目前只有"能举起来"的工程结果，没有新的学习机制，那更适合投 RSS/ICRA/IROS 或 workshop，而不是 ICLR。ICLR 审稿会直接问"方法层面的 novelty 是什么"。

### 3.2 建议的章节结构（重排）

1. **Abstract**（有真实数字后重写）
2. **1 Introduction**（补：为什么难 + 贡献列表 + 收窄的"first"宣称）
3. **2 Related Work**
   - 2.1 人形举重与整身负载交互（Ohmura2007, Arisumi2008, Kanoulas2018, Han2020, Zhang2023, Dao2023, Byrd2026, Zhang2026, Purushottam2025, Song2026）
   - 2.2 整身平衡与多接触控制（Kajita2004, Sentis2006/2010, Escande2014, Rocchi2015, Lee2015, Henze2016, Dantec2022）
   - 2.3 人形机器人体育技能（**新增**：badminton×2, CyboRacket, AdaPT, RoboNaldo, Marope, SKATER, archery/HuroCup）
   - 2.4 学习、运动模仿与 sim-to-real 人形控制（ExBody, HumanPlus, HOVER, AMO, 可选 AMP/PHC）
   - 2.5 外力/负载估计与自适应（**新增**：Rotella2015, SixthSense, Minimalist Compliance, LAC, ADAPT, FADA, Keemink2018）
   - 2.6 定位与差距（引用 HuroCup 和宇树演示视频说明"绝对首次"的限定）
4. **3 Problem Formulation**
   - 3.1 任务定义（选定变体、相位、成功/失败判据、负载协议）
   - 3.2 动力学与约束模型（含"双手+杠铃闭合链 → 力分配"的技术点）
   - 3.3 可行性分析（关节力矩 + CoP/ZMP 反推最大负载）
5. **4 Method**（具体化）
6. **5 仿真设置 / 6 实机设置**（G1 规格表引用 + H2 规格核实）
7. **7 Experiments**（指标、基线、消融、跨本体、失败与恢复）
8. **8 Limitations / 9 Conclusion**

### 3.3 方法提案（供填空，占位符处可落）

- **观测**：本体感知 + 关节力矩/电流 + IMU；可选深度相机。
- **负载表征**：在线估计杠铃质量与两手合成外力（基于动量动力学，引用 Rotella2015 / SixthSense）；或训练期注入随机负载质量，学一个负载隐变量 $z_t$（CVAE/编码器）。
- **策略**：相位条件化（phase label）+ 负载条件化的整身 RL 策略（$\pi_\theta(a_t|s_t,z_t,\phi_t)$），动作空间=关节位置/PD 目标或关节力矩；训练用多质量课程（curriculum over $m\in[0,m_{\max}]$）+ domain randomization（质量、摩擦、CoM 偏移）。
- **安全**：低层 WBC/MPC 跟踪 + ZMP/CoP 约束，或 CBF 安全滤波；头顶保持阶段用基于力矩裕度的降速/放弃决策。
- **Sim-to-real**：Isaac Gym/Lab + 系统辨识 + 可选 few-shot 动力学对齐（FADA）或残余策略微调。

### 3.4 实验与评测设计（建议指标表）

- 主指标：success rate（按 IWF 风格判据）；最大可稳定负载 $m_{\max}$（体重的 %）；overhead hold 时间。
- 过程指标：bar path 偏差（竖直/横向）、峰值 CoP 偏移、GRF 裕度、关节扭矩利用率、各相位时长。
- 消融：load-blind vs load-aware；无相位条件 vs 有；静态参考跟踪 vs 动态；sim-only vs sim-to-real；G1 vs H2 迁移。
- 鲁棒性：负载质量未知/偏移、摩擦变化、外部扰动、单手滑脱（接触损失）恢复。

### 3.5 写作与引用修正清单

1. 收窄"first"宣称并加 HuroCup + 宇树演示视频的限定说明。
2. 补 `Dantec2022` 到引用清单（正文已引用但清单缺失）。
3. 补 2.3 体育类和 2.5 外力/负载估计两节。
4. 投稿前逐个核对 arXiv 预印本的年份/作者/venue（Byrd2026、Zhang2026、Song2026、Purushottam2025、He2025 等）。
5. G1 规格用官方页引用；H2 规格核实后引用，不要写死参数。
6. 加一张方法架构图 + 一张结果图 + teaser 的规划。
7. 摘要写完实验后重写，填入真实数字，删掉全部 [占位符]。
8. 全文统一术语：抓举/挺举选一个变体并固定译法与缩写（snatch / power snatch / strict press）。

---

## 五、"全尺寸 + 符合人类举重规则 + 第一篇"宣称专项分析（2026-08-27 核验）

### 5.1 平台规格（官方页已核实）

| 平台 | 身高 | 体重 | 臂最大负载 | 臂/腿关节扭矩 | 是否"全尺寸" |
|---|---|---|---|---|---|
| G1 | 1.32 m | ~35 kg | ~2–3 kg | 臂 - / 膝 90/120 N·m | 否（中小型） |
| H1 | ~1.80 m | ~47 kg | 未公布额定（臂关节 ~75 N·m） | 膝 360 N·m | 是 |
| H1-2 | ~1.78 m | ~70 kg | 额定 ~7 kg / 峰值 ~21 kg | 臂 120 / 腿 360 N·m | 是 |
| H2 | 1.82 m | ~70 kg | 额定 ~7 kg / 峰值 ~15 kg | 臂 120 / 腿 360 N·m | 是 |

来源：unitree.com/g1、unitree.com/h1、unitree.com/H2（2026-08-27 抓取）。

### 5.2 "符合人类举重规则"的物理含义与风险

IWF 规则（snatch / clean & jerk）的关键物理要求：
- **正规杠铃杆**：男子 20 kg、女子 15 kg，杆长 220 cm，握距需落在杆上刻痕内；
- **连续上拉**、抓举一次到位；
- **头顶锁定并保持**（双臂伸直、双脚平踏、脚回到与躯干同一平面），**等裁判下放信号**后才可放杆；
- 除双脚外身体不得触台面。

对照平台能力：
- G1（臂 2–3 kg）→ 绝对无法举起 20/15 kg 正规杆，**"符合人类规则"对 G1 不成立**；
- H2（额定 7 / 峰值 15 kg/臂）→ 两手各持 10 kg（男杆）超过额定、接近峰值，且动态上拉阶段动载荷 ~1.5–2× 静载 → **男杆在额定意义下不现实**；**15 kg 女杆（每手 7.5 kg）更接近额定、相对可行**；
- 裁判下放信号、放杆、台上行为等**非物理环节机器人无法完整满足**，硬说"符合人类规则"会被审稿人逐条对表打脸。

### 5.3 结论

"全尺寸双足人形、能够符合人类举重规则的第一篇"**可以改成，但必须三处同时收紧**：

1. **"全尺寸"必须锚定 H2 / H1(-2)（≥1.78 m）**，G1 作辅助/小负载平台；若主平台是 G1，删除"全尺寸"。
2. **"符合人类规则"要么**：(a) 用 15 kg 女杆 + 一张"规则→可测约束"的符合性对照表（明确标注裁判/放杆等环节的改编）；要么 (b) 老实写"**IWF 启发（inspired）的协议**，保留连续上拉、锁定、保持、脚位等物理要求"。**绝不**在只用轻定制杆时声称"符合人类规则"。
3. **"第一篇"必须保留 `to our knowledge` + `peer-reviewed` 限定**：arXiv 空白 ≠ 全球空白（IEEE/中文期刊未穷尽；波士顿动力/Agility/优必选/宇树有力量与负载演示视频，虽非规则化举重研究）。

### 5.4 推荐写法（三选一）

- **A（最稳）**：*"To our knowledge, this is the first peer-reviewed study of autonomous, IWF-inspired barbell lifting on a full-size bipedal humanoid (Unitree H2)."*
- **B（强调规则符合性，需女杆+对照表支撑）**：*"We present the first full-size bipedal humanoid that can execute and stabilize a snatch-style overhead lift with a regulation barbell, supported by an explicit mapping from IWF rules to measurable robot constraints."*
- **C（最弱、最安全）**：*"…the first systematic study of autonomous competition-style lifting on full-size bipedal humanoids…"*

---

## 四、交付说明

- 本文档 = 锐评 + 二次调研 + 修改方案。
- 新文献已单独整理在下方"补充参考文献"清单中，可并入 `paper_sections_draft.md` 的 Citation checklist。
- 下一步建议：先定方法（Option A 还是 B）→ 补 Problem Formulation 与 Method 章节 → 再回填摘要。需要我直接按此方案改写 `paper_sections_draft.md` 请说一声。

---

## 补充参考文献（新增，附 arXiv 编号）

- [Christmann2022] Guilherme Christmann, Lin Yu-Ren, Rodrigo da Silva Guerra, Jacky Baltes. *Can a Robot Shoot an Olympic Recurve Bow? A preliminary study*. arXiv:2212.11071.
- [Chen2026Badminton] Yeke Chen et al. *Learning Human-Like Badminton Skills for Humanoid Robots*. arXiv:2602.08370.
- [Liu2025Badminton] Chenhao Liu et al. *Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning*. arXiv:2511.11218.
- [Ren2026CyboRacket] Peng Ren et al. *CyboRacket: A Perception-to-Action Framework for Humanoid Racket Sports*. arXiv:2603.14605.
- [Huang2026Tennis] Tao Huang et al. *Towards Professional Tennis Styles for Humanoid Robots (AdaPT)*. arXiv:2608.20087.
- [Zhong2026RoboNaldo] Yichao Zhong et al. *RoboNaldo: Humanoid Soccer Shooting via Motion-Guided Curriculum RL*. arXiv:2606.11092.
- [Wang2026Marope] Zihao Wang et al. *Cooperative Long Rope Skipping via Multi-Agent RL*. arXiv:2606.08064.
- [Gu2026SKATER] Junchi Gu et al. *SKATER: Roller Skating*. arXiv:2601.04948.
- [Rotella2015] Nicholas Rotella, Alexander Herzog, Stefan Schaal, Ludovic Righetti. *Humanoid Momentum Estimation Using Sensed Contact Wrenches*. arXiv:1507.04401 / HUMANOIDS 2015.
- [Chen2026SixthSense] Xingzhou Chen et al. *SixthSense: Task-Agnostic Proprioception-Only Whole-Body Wrench Estimation for Humanoids*. arXiv:2605.01427.
- [Shi2026Compliance] Haochen Shi et al. *Minimalist Compliance Control*. arXiv:2603.00913.
- [Liu2026LAC] Yang Liu et al. *LAC: Linear and Angular Compliance for Humanoid Whole-body Control*. arXiv:2608.25405.
- [Lyu2026ADAPT] Bofan Lyu et al. *ADAPT: Analytical Disturbance-Aware Policy Training*. arXiv:2606.16542.
- [Xie2026FADA] Angchen Xie et al. *FADA: Few-Shot Domain Adaptation via Dynamics Alignment*. arXiv:2606.28476.
- [Kim2025ARMADA] Jaehyung Kim et al. *ARMADA: A low-cost and lightweight 6 DoF bimanual arm for dynamic and contact-rich manipulation*. arXiv:2502.16908.
- [UnitreeG1] Unitree G1 specifications. https://www.unitree.com/g1/ (accessed 2026-08-27). 1.32 m, ~35 kg, 23 DoF, arm max load ~2-3 kg, knee peak torque 90/120 N·m.
- [UnitreeH1] Unitree H1 / H1-2 specifications. https://www.unitree.com/h1/ (accessed 2026-08-27). H1: ~1.80 m, ~47 kg, max joint torque 360 N·m; H1-2: ~1.78 m, ~70 kg, arm rated ~7 kg / peak ~21 kg.
- [UnitreeH2] Unitree H2 specifications. https://www.unitree.com/H2 (accessed 2026-08-27). 1.82 m, ~70 kg, 31 DoF, arm joint torque 120 N·m, arm max payload rated ~7 kg / peak ~15 kg.

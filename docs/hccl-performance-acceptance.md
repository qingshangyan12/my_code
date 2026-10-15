# HCCL 性能评审与验收方案

适用代码：

- 服务端实现：[cann/asc-devkit](https://gitcode.com/cann/asc-devkit) `impl/adv_api/detail/hccl`
- 客户端发起接口：`include/adv_api/hccl/hccl.h`

本文把一次集合通信从 AI Core 发起，到 AICPU 或 CCU 展开，再到网络搬运完成并被 kernel 感知，拆成可独立测量的时间段，并给出可执行的验收标准。数值阈值分为两类：

- **绝对阈值**：首轮在目标平台按链路规格标定后固化。文中给出的是工程参考量级，供评审确认，不是已签字的规格。
- **回归阈值**：相对已固化 baseline 不允许劣化。这类阈值可以直接进入门禁。

## 1. 目标

验收要回答五个问题：

1. AICPU 或 CCU 把一次通信请求展开成搬运指令，自身花了多少时间。
2. 从客户端通知服务端，到服务端真正开始搬运，固定延迟是多少。
3. 各算法模板在目标拓扑和数据量下，带宽效率是否达到链路能力。
4. 通信与计算重叠时，双方互相拖慢的比例是否可接受。
5. `hccl.h` 客户端在 AI Core 上的头开销有多大。这段时间占核，不能被通信掩盖。

最终交付是一张可门禁的验收表：每个指标有唯一定义、采集点、统计口径（P50/P99）、平台和通过条件。

## 2. 代码路径

一次通信的完整路径如下。

```text
AI Core kernel                         服务端                         网络
hccl.h / Hccl<serverType>
  InitV2
  Prepare  ──写 GM 参数 + dcci──┐
  Commit   ──写 commit region──┼──► AICPU: aicpu_kfc
  （计算可与通信重叠）          │      selector → executor → ins_temp_*
  Wait     ◄──读 finish/wait───┤      下发 SDMA / RDMA / notify
  Finalize                     │
                               └──► CCU: ccu_kernel_kfc_server
                                      ccu_kernel_* 直接搬运
                                      mesh / NHR / multi_jetty / mem2mem / 2die
```

客户端声明在 `include/adv_api/hccl/hccl.h`，实现按架构分发：

| 架构 | 实现 |
|---|---|
| `__NPU_ARCH__ == 2201` | `impl/platform_v220/hccl_aicpu.h` |
| `9201 / 9202` | `impl/platform_v310/hccl_aicpu.h`、`impl/platform_v960/hccl_ccu_v0.h` |
| `3510` | `impl/platform_v310/hccl_aicpu.h`、`impl/platform_v310/hccl_ccu_v0.h` |

每个 Prepare 类接口在 `impl/hccl_impl.h` 外包了一层 `DFX_MAKE_GUARD`（`common/hccl_impl_dfx.h`），再进入 `impl_`。

服务端目录：

| 路径 | 职责 |
|---|---|
| `cc/src/aicpu_kfc/` | AICPU 通信框架、资源、transport |
| `cc/src/ops/*/selector` | 按拓扑和数据量选模板 |
| `cc/src/ops/*/executor` | 编排资源与执行 |
| `cc/src/ops/*/template/aicpu/ins_temp_*` | AICPU 算法展开 |
| `cc/src/ops/kfc_server/template/ccu/kernel/ccu_kernel_kfc_server` | CCU 常驻服务 |
| `cc/src/ops/*/template/ccu/kernel/ccu_kernel_*` | CCU 搬运 kernel |
| `common/kfc_server_message.h` | KFC 消息与 commit/wait region |
| `aicpu_kfc/common/debug/profiling/inc/adapter_prof.h` | msprof 打点 |

KFC 协议常量（`kfc_server_message.h`）：

- `KFC_MAX_MISSION_NUM = 2`
- 每个 mission 两个 signal region：`KFC_COMMIT_REGION_INDEX = 0`，`KFC_WAIT_REGION_INDEX = 1`
- 客户端通过 `GetCommitCkeAddr` / `GetWaitCkeAddr` 定位这两个 region

已覆盖的集合通信与模板包括 AllGather、ReduceScatter、AllReduce、AlltoAll、AlltoAllV、BatchWrite、AlltoAllvWrite；模板形态包括 mesh_1D、NHR、2die、multi_jetty、omnipipe、Z_axis_detour、dpu、mem2mem、meshchunk。平台实现分 `platform_v220`、`platform_v310`、`platform_v960`。

## 3. 测量原则

1. **时钟**。跨 AI Core、AICPU、CCU 的延迟统一用 syscnt。首轮标定各主体读数的固定偏差，偏差计入测量不确定度，不计入被测时间。AI Core 本核头开销只用本核 syscnt，不需要跨主体对时。
2. **统计**。每个指标报 P50 和 P99。P99 用于延迟类门禁，P50 用于带宽类门禁。样本不少于 1000 次稳态迭代，丢掉前 50 次预热。
3. **开销归属不能重复计算**。客户端把参数 flush 到 HBM 的时间算在客户端头开销里；服务端从 HBM 可见到发出首条搬运，算在 KFC 前向延迟里。两端打点以“指令发出”和“对端首次观察到”分开记录。
4. **打点扰动**。profiling 打开相对关闭，端到端变化必须小于 1%，否则该组数据无效。
5. **小包和大包分开验收**。固定开销 α 用小包暴露，带宽 β 用大包暴露。只用大包带宽做门禁会掩盖展开和客户端劣化。

端到端固定开销定义为四段之和：

```text
α = T_client + T_kfc_fwd + T_expand_first + T_kfc_back
```

- `T_client`：一次 Prepare + 一次 Commit + Wait 进入首次轮询（第 8 节 F2+F4+F5）
- `T_kfc_fwd`：commit region 对服务端可见，到服务端发出首条搬运（第 5 节 B1）
- `T_expand_first`：服务端内部到首条搬运指令的时间（第 4 节 A1/A2 的首指令口径）
- `T_kfc_back`：最后一笔数据落地，到客户端 Wait 观察到完成（第 5 节 B2，含轮询粒度）

α 的直接测量值与四段之和的相对差必须小于 10%。差更大说明有未归类开销（cache 一致性、CKE 写未合并、跨核同步）。

## 4. 维度一：AICPU / CCU 展开时间

展开时间是服务端内部耗时，起点是服务端取到一条有效通信请求，终点按两个口径分别统计。

| 口径 | 终点 | 用途 |
|---|---|---|
| 首指令 | 发出第一条搬运指令 | 决定小包延迟，进入 α |
| 全量展开 | 本条请求的全部 task / 指令构造完成 | 判断展开能否被传输时间掩盖 |

AICPU 路径再拆三段，便于定位：

| 段 | 区间 |
|---|---|
| T_sel | kfc server 取到消息 → selector 返回模板 |
| T_exec | selector 返回 → executor 完成资源编排 |
| T_tmpl | 进入 `ins_temp_*` → 最后一个 transport task 下发 |

CCU 路径的展开区间是 `ccu_kernel_kfc_server` 收到 mission，到对应 `ccu_kernel_*` 发出第一条搬运。CCU 没有 host/AICPU 侧的 task 循环，全量展开通常远小于 AICPU，仍要单列，避免把“CCU 更快”写成不可回归的印象。

采集：

- AICPU 复用 `hrtMsprofSysCycleTime` / `hrtMsprofReportApi`，打在 selector、executor、template 出入口，受 profiling 开关控制。
- CCU 用 GM ring buffer 记 `(event_id, syscnt)`，host 离线对齐。

验收关注：P99 绝对值、首指令与全量展开的比例、随 rank 数和 repeat 的斜率。斜率超线性即失败，即使绝对值仍在阈值内。

## 5. 维度二：KFC 通信展开延迟

这是客户端和服务端之间的通知延迟，不含算法搬运本身。

| 指标 | 起点 | 终点 |
|---|---|---|
| B1 前向 | commit region 写入并对服务端可见 | 服务端发出首条搬运指令 |
| B2 回程 | 最后一笔数据落地 | Wait 读到完成（finishCnt 或 wait region） |

B1 包含服务端轮询周期和唤醒。轮询实现若是固定 sleep 或稀疏轮询，P99 会被轮询间隔主导，应把轮询间隔单独报出来。

B2 与客户端 Wait 的单次轮询代价（F5）连在一起看：服务端置位很快，但客户端很久才看见，算 B2 失败，不算展开失败。

协议对应 `KFC_COMMIT_REGION_INDEX` 与 `KFC_WAIT_REGION_INDEX`。并行 Mesh+NHR 等一个 mission 内多阶段的模板，按阶段分别记 B1/B2，不能只记整次 AllGather。

## 6. 维度三：算法搬运网络性能

只度量数据搬运阶段，展开和 KFC 延迟已在前两节扣除。

### 6.1 测试矩阵

| 维度 | 取值 |
|---|---|
| 算子 | AllGather、ReduceScatter、AllReduce、AlltoAll、AlltoAllV |
| 模板 | mesh_1D、NHR、2die、multi_jetty、omnipipe、Z_axis_detour、dpu、mem2mem |
| 数据量 | 8KB 到 1GB，按 2 的幂 |
| 规模 | 单机 rank 数、多机 rank 数，覆盖目标拓扑 |
| 平台 | v220、v310、v960 |
| 参照 | 同配置 host 下发的 standalone HCCL 单算子 |

### 6.2 指标

- **算法带宽** `alg_bw = 有效数据量 / 搬运阶段时间`
- **总线带宽**按集合通信系数折算。AllReduce、AllGather、ReduceScatter 使用 `bus_bw = alg_bw × 2(n−1)/n`，n 为 rank 数。AlltoAll 按每 rank 发出的数据量计，不套这个系数。
- **带宽效率** = `bus_bw / 链路理论峰值`
- **拐点** = 带宽效率第一次达到 90% 峰值时的最小消息长度

带宽-消息长度曲线每个模板保留一份。版本对比看整条曲线，不只看最大包。

### 6.3 选择器

`ops/*/selector` 自动选中的模板，性能必须达到该矩阵里实测最优模板的 98%。否则选择器验收失败，即使被选模板自身带宽达标。

## 7. 维度四：通信与计算重叠

单独跑和重叠跑使用同一份计算 kernel、同一份通信参数、同一组核。

| 符号 | 含义 |
|---|---|
| `T_comp_solo` | 只跑计算 |
| `T_comm_solo` | 只跑通信 |
| `T_comp_overlap` | 重叠时计算侧耗时 |
| `T_comm_overlap` | 重叠时通信侧耗时 |
| `T_fused` | 融合算子端到端 |

```text
ρ_comp = T_comp_overlap / T_comp_solo − 1
ρ_comm = T_comm_overlap / T_comm_solo − 1
E      = (T_comp_solo + T_comm_solo − T_fused) / min(T_comp_solo, T_comm_solo)
```

`ρ` 是被对方拖慢的比例，`E = 1` 表示较短的一方被完全掩盖。

负载分三组，因为干扰源不同：

| 计算 | 预期争用 |
|---|---|
| cube 密集 matmul | 低 HBM 争用，作为重叠收益上限 |
| vector 密集 | 中等 |
| 访存密集 | 与 SDMA 抢 HBM，作为干扰上限 |

通信再分片内 SDMA 与跨机 RDMA。CCU 形态额外测 `ccu_kernel_kfc_server` 常驻核占用对计算可用核数和 HBM 带宽的影响，把常驻核从计算核集合中扣除后再算 `ρ_comp`，避免把“少了一核”算成缓存颠簸。

客户端 Prepare/Commit 插在计算流水中的时间属于第 8 节，不计入 `ρ_comp`。

## 8. 维度五：客户端头开销

接口在 `include/adv_api/hccl/hccl.h`。典型顺序是 Init、Prepare、Commit、Wait、Finalize。`commit = true` 的 Prepare 会把 T2 并进同一次调用，验收时两种调用方式都要测。

这段时间跑在 AI Core 上，MC2 里它位于计算间隙，不能被通信掩盖，所以不并入 KFC 延迟。

### 8.1 分段

| 段 | 代码行为 | 特征 |
|---|---|---|
| T0 `InitV2` + `SetCcTilingV2` | 读 context，解析 msg area（`MAX_DCCI_CNT`、finishCnt 偏移） | 每个 kernel 一次 |
| T1 Prepare | 组 `CommonPrepareParam`，`WriteHBMData` 写 GM，按 cache line `FlushDataCache` | 每个 handle 一次。AlltoAllV 在 v310 CCU 实现里按 `rankNum` 循环刷 cache line，随 rank 线性增长 |
| T2 `Commit` | 更新 `handleCommitCnt_` / `globalCurCommitCnt_`，写 commit CKE，再 flush | 每个 repeat 一次，目标是常数 |
| T3 Wait 入口 | 进入轮询前的准备 | 常数 |
| T4 Wait 单次轮询 | 读 finishCnt GM 并 invalidate | 决定完成可见粒度 |
| T5 多核 | 仅 `workingFlag_` 为真的核写 msg area（`GetBlockIdx() == config.blockId`，并可限定 AIC 或 AIV） | 非 working core 的自旋不计入头开销；working core 的串行段会计入所有核的等待 |

`DFX_MAKE_GUARD` 单独开关。关闭时应被编译掉；打开时的增量单列，不混进 T1。

### 8.2 测法

1. **本核 syscnt**。API 前后读 cycle，写入 GM ring buffer。GM 写异步，打点口径是指令发出；flush 完成时刻另记一条，供和第 5 节对账。
2. **指令级 L0**。这些 API 都是 `__aicore__ inline`。用仿真器或 `tests/api/adv_api/hccl/ccu/stub` 统计每个 API 的指令数，作为无硬件回归。头开销退化几乎都表现为指令膨胀。
3. **空服务端微基准**。服务端 stub 成收到即完成。端到端时间减去已测的 B1、B2，应等于 T_client。与打点法互查。
4. **扫描**。算子 × rank 数 × repeat × AICPU/CCU × DFX 开/关。AlltoAllV 必须扫 rank 数，拟合线性项斜率。

## 9. 打点实现约定

| 位置 | 手段 | 关闭时行为 |
|---|---|---|
| AICPU selector / executor / template | `adapter_prof.h` 的 msprof | 开关关闭则不调用上报 |
| CCU kernel、KFC server | GM ring buffer `(event_id, syscnt)` | 编译宏裁掉 |
| AI Core 客户端 | 本核 syscnt + 同一条 GM ring buffer | 编译宏裁掉 |
| stub | 统计指令数或模拟 cycle | 始终可用，供 L0 |

事件号固定枚举，至少包含：`CLIENT_PREPARE_BEGIN/END`、`CLIENT_COMMIT_ISSUE`、`CLIENT_COMMIT_VISIBLE`、`CLIENT_WAIT_ENTER`、`CLIENT_WAIT_OBSERVE`、`SERVER_MSG_TAKEN`、`SERVER_SELECT_DONE`、`SERVER_EXEC_DONE`、`SERVER_FIRST_MOVE`、`SERVER_EXPAND_DONE`、`SERVER_DATA_DONE`、`SERVER_SIGNAL_SET`。

同一条请求用 `handleId + repeatIndex` 关联两端事件。并行 mission 再加 `missionIndex`。

## 10. 分层测试

| 层级 | 环境 | 内容 | 何时跑 |
|---|---|---|---|
| L0 | stub / 仿真，无硬件 | 客户端 API 指令数；AICPU/CCU 展开路径指令数或模拟 cycle | 每个 PR |
| L1 | 单机真机 | T_expand、T_kfc、客户端微基准、各模板带宽曲线 | 每日 |
| L2 | 单机真机 | MC2 融合算子的 ρ、E，以及客户端占核比 | 每日 |
| L3 | 多机 | RDMA 带宽效率、rank 扩展线性度 | 每个版本 |

L0 失败阻塞合入。L1/L2 相对 baseline 劣化超过回归阈值阻塞合入。L3 失败阻塞发版，不阻塞每日合入。

## 11. 验收标准

### 11.1 绝对阈值（首轮标定后固化）

| 编号 | 指标 | 参考量级 | 层级 |
|---|---|---|---|
| A1 | AICPU 首指令展开 P99 | ≤ 5μs | L1 |
| A1b | AICPU 全量展开 P99 | ≤ 10μs，且相对 rank 数不超线性 | L1 |
| A2 | CCU 首指令展开 P99 | ≤ 1μs | L1 |
| A2b | CCU 全量展开 P99 | ≤ 2μs | L1 |
| B1 | KFC 前向 P99 | ≤ 5μs | L1 |
| B2 | KFC 回程 P99 | ≤ 5μs | L1 |
| B3 | 小包 α | AICPU 路径 ≤ 20μs；CCU 路径 ≤ 5μs | L1 |
| C1 | 大包带宽效率 | 片内 SDMA ≥ 90% 理论峰值；跨机 RDMA ≥ 85% | L1 / L3 |
| C2 | 带宽拐点 | 达到 90% 峰值的最小消息长度不随版本变大 | L1 |
| C3 | 相对 standalone HCCL | 同配置 ≥ 参照的 95% | L1 |
| C4 | selector | ≥ 同矩阵实测最优模板的 98% | L1 |
| D1 | ρ_comp | cube ≤ 5%；访存密集 ≤ 10% | L2 |
| D2 | ρ_comm | ≤ 10% | L2 |
| D3 | 重叠效率 E | ≥ 80% | L2 |
| F1 | InitV2 + SetCcTilingV2 | ≤ 5μs | L1 |
| F2 | 定长算子单次 Prepare | ≤ 1μs；flush 次数有上界且不随数据量变化 | L0 / L1 |
| F3 | AlltoAllV Prepare 斜率 | 每 rank ≤ 50ns；64 rank 合计 ≤ 4μs | L1 |
| F4 | 单次 Commit | ≤ 0.5μs | L0 / L1 |
| F5 | Wait 单次轮询 | 完成感知额外延迟 ≤ 1μs，与 B2 联合判定 | L1 |
| F6 | DFX guard | 关闭时指令增量为 0；打开时单 API 耗时增量 < 10% | L0 |
| F7 | 客户端占核比 | Prepare+Commit+Wait 入口合计 ≤ 端到端的 1%（大包）或 5%（小包） | L2 |

A、B、F 的微秒数是评审用起点。标定规则：取首轮稳定版本的 P99，与上表取较严的一个，作为该平台的签字阈值。若首轮 P99 高于上表，必须有代码路径说明，不能直接把阈值放宽到实测值。

### 11.2 回归阈值

| 编号 | 条件 | 动作 |
|---|---|---|
| E1 | L0 任一 API 或展开路径指令数相对 baseline 波动超过 ±5% | 阻塞合入 |
| E2 | L1/L2 任一指标劣化超过 3% | 阻塞合入 |
| E3 | profiling 打开后端到端变化 ≥ 1% | 该次数据作废，先修打点 |
| E4 | α 直接测量与分段之和相对差 ≥ 10% | 不得宣布 α 达标，先补事件 |

带宽类看 P50，延迟类看 P99。只改善 P50、P99 变差，判失败。

## 12. 验收流程

1. 三个平台各跑全矩阵，产出 baseline：带宽曲线、α 分段、指令数。
2. 架构评审确认第 11.1 节的绝对阈值，按平台写入规格表。未确认前，只启用第 11.2 节的回归阈值。
3. L0 接入 PR 门禁。L1/L2 出每日报表，超阈值告警并阻塞。
4. 新增模板或新算子必须补齐矩阵中对应的一行，否则 C4 无法判定，视为未验收。
5. 客户端 `hccl.h` 的接口签名或 msg area 布局变更，必须重跑 F 组与 B 组。服务端模板变更必须重跑 A 组与 C 组。MC2 融合算子变更必须重跑 D 组与 F7。

## 13. 评审时需要拍板的事项

| 项 | 建议 | 需要确认 |
|---|---|---|
| 绝对阈值是否按第 11.1 节的量级签字 | 是，允许按平台写成三列 | 平台负责人 |
| standalone HCCL 参照的版本 | 与本仓库同一发布列车的 host HCCL | 通信栈负责人 |
| 理论峰值取哪条链路 | 片内取 SDMA 规格，跨机取网卡单向有效带宽 | 硬件规格 |
| CCU 常驻核是否从计算核数中扣除 | 扣除后再算 ρ_comp，并在报告里同时给出不扣除的数 | MC2 算子负责人 |
| AlltoAllV 的 64 rank 是否覆盖目标最大规模 | 若产品规模更大，F3 按最大 rank 重算 | 产品规划 |

## 14. 不在本方案内

- 集合通信的数值正确性。正确性沿用现有功能测试，性能数据只在正确性通过的运行上统计。
- Host 图模式调度（`ops/interface_graph_mode`）的框架调度时延。若后续要验收图模式，单列一张表，不混入 α。
- 拥塞、丢包、链路故障下的性能。本方案只覆盖健康链路。

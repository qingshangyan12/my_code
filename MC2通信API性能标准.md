# MC2 通信 API 性能标准

AIC 侧通信开销。Client 用 ticks，KFC 用 us，纯通信与 HCCL 持平。验证环境：8P fullmesh · 4K · 30 轮。

本标准只覆盖 MC2 通信 API 的软件开销，以及纯通信相对 HCCL 的持平要求。MC2 算子计算部分不给性能标准。三段分别打点、分别达标，不合成总分。`sq_expand(miss)` 仅参考，不做强校验。

- [1. 总则](#1-总则)
- [2. Client 头开销](#2-client-头开销)
- [3. KFC Server 展开开销](#3-kfc-server-展开开销)
- [4. 纯通信](#4-纯通信)
- [5. 验收判定](#5-验收判定)
- [6. 1130 目标](#6-1130-目标)

---

## 1. 总则

### 1.1 范围

对象是 MC2 场景下的通信 API 开销，不是 MC2 融合算子的 MatMul/计算性能。适用 core 为 `aic`。primitive 分两档：

- `a2a/ag/ar/rs` 共用一档（AlltoAll / AllGather / AllReduce / ReduceScatter）
- `a2av` 单独一档（AlltoAllV）

`a2av` 仅 Prepare、Commit、Wait、msg_parse、sq_expand 与第一档不同；Client Init、Finalize 与 Server Init 与第一档相同。

### 1.2 验证环境

| 项 | 要求 |
|---|---|
| 拓扑 | 8P fullmesh |
| 主频 | AICore 1650 MHz；AICPU 1500 MHz |
| 测试数据量 | 4K |
| 统计方式 | 一轮 = 1 次冷启动 + 20 次 warmup + 100 次执行；共执行 30 轮 |
| 热路径样本 | 每轮 warmup 不计入达标集；mean / p50 / p99 取 30 × 100 次执行 |
| 冷启动样本 | 每轮 1 次冷启动，用于观察 Client/Server Init 与 sq_expand(miss)；miss 仅参考 |

### 1.3 三段定义

| 阶段 | 名称 | 测什么 | 单位 | 达标方式 |
|---|---|---|---|---|
| 1 | Client 头开销 | Client Init / Prepare / Commit / Wait / Finalize | ticks | 每个子项 mean、p50、p99 均不超过本标准表 |
| 2 | KFC Server 展开开销 | Server Init / msg_parse / sq_expand(hit) / sq_expand(miss) | us | Init、msg_parse、sq_expand(hit) 强校验；sq_expand(miss) 仅参考 |
| 3 | 算法执行开销 | 纯通信耗时，不含 MC2 计算 | 与 HCCL 对比 | 与 HCCL 同等通信持平；数值对照 UBX 算法选择建议 |

### 1.4 计时规则

三段分开统计，各有打点，互不包含、互不折算。阶段 1 的 Wait 是通信已经收到之后、把数据写入共享内存，直到 Wait 执行完成的耗时。

Client Init 与 Finalize 是算子粒度：一次 MC2 算子执行只调用一次。Server Init 是 AICPU 通信算子初始化，同样按算子粒度一次。Prepare / Commit / Wait 按实际打点样本统计 mean / p50 / p99。

`sq_expand` 只报 hit、miss 两条时延。hit 强校验，miss 仅作参考，不做强校验。

### 1.5 子项边界

| 子项 | 阶段 | 边界 |
|---|---|---|
| Client Init | 1 | Client 侧初始化，一次 MC2 算子执行只发生一次 |
| Prepare | 1 | 通信请求的数据准备 |
| Commit | 1 | 提交通信请求 |
| Wait | 1 | 通信已收到之后，把数据写入共享内存，直到 Wait 执行完成 |
| Finalize | 1 | Client 侧收尾清理，一次 MC2 算子执行只发生一次 |
| Server Init | 2 | KFC Server / AICPU 通信算子初始化，一次 MC2 算子执行只发生一次 |
| msg_parse | 2 | KFC 收到 Client 消息后的读取与解析 |
| sq_expand(hit) | 2 | SQE 展开且命中 cache；强校验 |
| sq_expand(miss) | 2 | SQE 展开且未命中 cache；仅参考，不强校验 |
| 纯通信 | 3 | 集合通信本身的执行时间，验收标准为与 HCCL 通信持平 |

---

## 2. Client 头开销

Client 头开销在 AIC 上统计，单位是 ticks。表内数值为上限：实测越小越好，mean / p50 / p99 都不得超过对应格子。a2av 的 Client Init、Finalize 与 a2a/ag/ar/rs 相同。

### 2.1 a2a / ag / ar / rs

| metric | mean | p50 | p99 | 单位 | 调用次数 |
|---|---:|---:|---:|---|---|
| Client Init | 305 | 303 | 445 | ticks | 每个 MC2 算子执行 1 次 |
| Prepare | 872 | 876 | 1075 | ticks | 按打点样本 |
| Commit | 608 | 605 | 710 | ticks | 按打点样本 |
| Wait | 893 | 893 | 901 | ticks | 按打点样本 |
| Finalize | 1402 | 1364 | 1854 | ticks | 每个 MC2 算子执行 1 次 |

### 2.2 a2av

| metric | mean | p50 | p99 | 单位 | 调用次数 |
|---|---:|---:|---:|---|---|
| Client Init | 305 | 303 | 445 | ticks | 每个 MC2 算子执行 1 次，与 a2a/ag/ar/rs 相同 |
| Prepare | 2205 | 2216 | 2389 | ticks | 按打点样本 |
| Commit | 621 | 620 | 718 | ticks | 按打点样本 |
| Wait | 902 | 903 | 909 | ticks | 按打点样本 |
| Finalize | 1402 | 1364 | 1854 | ticks | 每个 MC2 算子执行 1 次，与 a2a/ag/ar/rs 相同 |

### 2.3 Wait 口径

使用互斥的 `wait_sw`。起点：通信数据已经收到，开始把数据写入共享内存。终点：Wait 执行完成。不含等待通信完成的时间，不含 KFC 展开，不含纯通信搬数。

### 2.4 Client 热路径 mean 上限（ticks）

| metric | a2a/ag/ar/rs mean | a2av mean |
|---|---:|---:|
| Prepare | 872 | 2205 |
| Commit | 608 | 621 |
| Wait | 893 | 902 |

Client Init / Finalize 两档相同，未列入本对照。

---

## 3. KFC Server 展开开销

KFC Server 展开开销单位是 us。Server Init、msg_parse、sq_expand(hit) 强校验；sq_expand(miss) 仅参考。本标准只卡时延，不规定 cache key。a2av 的 Server Init 与 a2a/ag/ar/rs 相同。

### 3.1 a2a / ag / ar / rs

| metric | mean | p50 | p99 | 单位 | 校验 |
|---|---:|---:|---:|---|---|
| Server Init | 7.6 | 7.6 | 8.9 | us | 强校验 |
| msg_parse | 1.9 | 1.5 | 4.7 | us | 强校验 |
| sq_expand(hit) | 70.3 | 67.8 | 83.7 | us | 强校验 |
| sq_expand(miss) | 145.0 | 138.8 | 167.0 | us | 仅参考 |

### 3.2 a2av

| metric | mean | p50 | p99 | 单位 | 校验 |
|---|---:|---:|---:|---|---|
| Server Init | 7.6 | 7.6 | 8.9 | us | 强校验 |
| msg_parse | 2.9 | 2.5 | 7.5 | us | 强校验 |
| sq_expand(hit) | 72.4 | 70.2 | 78.1 | us | 强校验 |
| sq_expand(miss) | 148.0 | 147.0 | 177.0 | us | 仅参考 |

### 3.3 sq_expand mean 上限（us）

| primitive | sq_expand(hit) mean · 强校验 | sq_expand(miss) mean · 仅参考 |
|---|---:|---:|
| a2a/ag/ar/rs | 70.3 | 145.0 |
| a2av | 72.4 | 148.0 |

### 3.4 Server Init

Server Init 是 AICPU 上通信算子的初始化，与 Client Init 分开打点。一次 MC2 算子执行只做一次。各 primitive 共用同一 Server Init 门限。

---

## 4. 纯通信

阶段 3 只计纯通信。算法执行开销是集合通信本身的执行时间，不含 Client 头开销、不含 KFC 展开、不含 MC2 算子里的计算。

验收标准：与 HCCL 同等通信持平。具体数值对照 [UBX 算法选择建议（OneBox）](https://onebox.huawei.com/v/2a6521d10248ac030a6077535cd5fa2d?type=1&sheet=UBX%20%E7%AE%97%E6%B3%95%E9%80%89%E6%8B%A9%E5%BB%BA%E8%AE%AE) 中的 HCCL 通信基准。

这里给的是通信 API 开销。MC2 算子的 MatMul 等计算性能不在此文件提供标准，也不把融合墙钟算进阶段 1 或阶段 2。

| 要对照的 | 不要算进来的 |
|---|---|
| 同拓扑、同数据量下 HCCL 纯通信时延/带宽 | Client Init / Prepare / Commit / Wait / Finalize |
| UBX 表中对应 primitive 的 HCCL 基准 | Server Init / msg_parse / sq_expand |
| AlltoAll / AllGather / AllReduce / ReduceScatter / AlltoAllV 的通信实现 | MC2 融合计算、重叠后的算子 E2E |

---

## 5. 验收判定

### 5.1 判定规则

1. 阶段 1、2、3 分别验收。强校验项任一不达标，该场景判定不通过。不得用其他阶段的余量抵消。
2. 阶段 1：Client Init / Prepare / Commit / Wait / Finalize 的 mean、p50、p99 均不超过表内上限。
3. 阶段 2：Server Init / msg_parse / sq_expand(hit) 的 mean、p50、p99 均不超过表内上限。sq_expand(miss) 只记录、只对比参考值，超标不判失败。
4. 阶段 3：纯通信与 HCCL 持平；数值对照 UBX 算法选择建议。
5. 单位必须一致：Client 用 ticks，KFC 用 us。不得换算后和另一张表比。
6. 必须在本标准规定的验证环境下测：8P fullmesh，AICore 1650 MHz，AICPU 1500 MHz，数据量 4K；一轮 = 1 次冷启动 + 20 次 warmup + 100 次执行，共 30 轮。热路径用 30 × 100 次执行样本。
7. Client Init / Finalize / Server Init 按每个 MC2 算子执行一次的样本统计；其余强校验子项按热路径打点统计。

### 5.2 验收清单

| 场景 | 强校验 | 仅参考 |
|---|---|---|
| AIC · a2a/ag/ar/rs | Client：Init、Prepare、Commit、Wait、Finalize；KFC：Server Init、msg_parse、sq_expand(hit)；纯通信与 HCCL 持平 | sq_expand(miss) |
| AIC · a2av | Client：Init、Prepare、Commit、Wait、Finalize；KFC：Server Init、msg_parse、sq_expand(hit)；纯通信与 HCCL 持平 | sq_expand(miss) |

### 5.3 未列入本标准

MC2 算子计算性能；Client 与 KFC 的合成总分；sq_expand 的 cache key；AIV 作为 core 的软件表（本表 core 仅为 aic）；sq_expand(miss) 的失败门限。

---

## 6. 1130 目标

本章是 1130 要达到的目标，不替代前面的现行标准。计时口径、验证环境、三段划分与前面章节相同。1130 只给单值，按 mean 上限验收；p50 / p99 继续上报，不按分位强卡。sq_expand(miss) 仍仅参考。纯通信仍与 HCCL 持平。

### 6.1 Client · 1130 mean 上限（ticks）

| primitive | Client Init | Prepare | Commit | Wait | Finalize |
|---|---:|---:|---:|---:|---:|
| a2a / ag / ar / rs | 200 | 400 | 300 | 600 | 1000 |
| a2av | 200 | 1100 | 300 | 600 | 1000 |

Client Init / Finalize 与 a2a/ag/ar/rs 相同。单位 ticks。每个 MC2 算子执行 Init、Finalize 各 1 次。

### 6.2 Server · 1130 mean 上限（us）

| primitive | Server Init | msg_parse | sq_expand(hit) | sq_expand(miss) |
|---|---:|---:|---:|---:|
| a2a / ag / ar / rs | 5 | 1.3 | 40 | 120 |
| a2av | 5 | 2.0 | 40 | 120 |

Server Init、msg_parse、sq_expand(hit) 强校验；sq_expand(miss) 仅参考。Server Init 两档相同。

### 6.3 相对现行 mean

| 阶段 | metric | a2a/ag/ar/rs 现行 mean | 1130 | a2av 现行 mean | 1130 |
|---|---|---:|---:|---:|---:|
| Client (ticks) | Client Init | 305 | 200 | 305 | 200 |
| Client (ticks) | Prepare | 872 | 400 | 2205 | 1100 |
| Client (ticks) | Commit | 608 | 300 | 621 | 300 |
| Client (ticks) | Wait | 893 | 600 | 902 | 600 |
| Client (ticks) | Finalize | 1402 | 1000 | 1402 | 1000 |
| Server (us) | Server Init | 7.6 | 5 | 7.6 | 5 |
| Server (us) | msg_parse | 1.9 | 1.3 | 2.9 | 2.0 |
| Server (us) | sq_expand(hit) | 70.3 | 40 | 72.4 | 40 |
| Server (us) | sq_expand(miss) 参考 | 145.0 | 120 | 148.0 | 120 |

Client mean 对照（ticks）：

| metric | a2a/ag/ar/rs 现行 mean | a2a/ag/ar/rs 1130 | a2av 现行 mean | a2av 1130 |
|---|---:|---:|---:|---:|
| Init | 305 | 200 | 305 | 200 |
| Prepare | 872 | 400 | 2205 | 1100 |
| Commit | 608 | 300 | 621 | 300 |
| Wait | 893 | 600 | 902 | 600 |
| Finalize | 1402 | 1000 | 1402 | 1000 |

sq_expand(hit) mean 对照（us）：

| primitive | 现行 mean | 1130 目标 |
|---|---:|---:|
| a2a/ag/ar/rs | 70.3 | 40 |
| a2av | 72.4 | 40 |

hit 强校验；miss 仅参考，未列入本对照。

### 6.4 1130 验收

1. 强校验项的 mean 不超过本章目标值，该场景才通过 1130。不得用其他阶段余量抵消。
2. Client：Init 200、Prepare 400（a2av 为 1100）、Commit 300、Wait 600、Finalize 1000，单位 ticks。
3. Server：Init 5、msg_parse 1.3（a2av 为 2.0）、sq_expand(hit) 40，单位 us。sq_expand(miss) 参考 120 us，超标不判失败。
4. 纯通信与 HCCL 持平，与前面章节相同。
5. 验证环境、打点边界、统计轮次与前面章节相同。

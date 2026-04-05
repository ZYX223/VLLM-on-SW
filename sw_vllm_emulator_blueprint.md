# SW vLLM Emulator Blueprint

## 当前仓库落点

当前仓库以 `VLLM-on-SW` 为主线，并已经把 `SW emulator` 作为一个可切换 attention backend 原型集成进来。

因此后续开发策略不是再单独维护一个平行仓库，而是继续基于这里的 `prefill / decode / scheduler / KV cache` 主干扩展 `sw_emulator` 后端能力。

## 目标

这个项目的合理定位不是“真实神威性能模拟器”，而是：

- `vLLM-on-SW` 的开发代理层
- 神威后端的功能原型
- 神威编程模型的验证环境

核心目标：

1. 在没有真实神威环境时，先把 `vLLM` 风格的推理主干跑通。
2. 先稳定神威后端接口，再逐步替换底层 kernel。
3. 用 emulator 判断优化方向，不直接外推真实神威性能。

## 当前已落地内容

- `LLM(..., backend="sw_emulator")` 已接入配置层。
- attention 路径已抽象为 backend dispatch，支持 `default` 与 `sw_emulator` 两种模式。
- `sw_emulator` 目前覆盖：
  - KV cache 写入
  - prefill attention
  - prefix cache 场景
  - decode attention
- 增加了语义单测，校验 paged KV cache 和 causal attention 行为。

## 分阶段计划

### 阶段 1：稳定 attention backend 抽象

目标：

- 保持 `default` backend 行为不回退
- 让 `sw_emulator` 成为一个稳定、可扩展的语义后端入口
- 补齐针对 backend 选择和回归行为的测试

### 阶段 2：扩大后端覆盖面

优先顺序：

1. GEMM
2. RMSNorm
3. RoPE
4. attention prefill
5. attention decode
6. KV cache 读写与布局

原则：

- 先用 reference 语义实现把接口跑通
- 再逐步替换成更接近神威执行模型的实现
- 不为了一时方便把接口设计成 CUDA 特有语义

### 阶段 3：靠近推理执行模型

目标：

- 从“单算子正确”推进到“按框架方式运行”

任务：

1. 引入更明确的 KV cache manager 语义
2. 沉淀 page / block 抽象
3. 区分 prefill / decode 的 backend 行为边界
4. 在 scheduler 流程上增加更强的端到端回归用例

### 阶段 4：靠近神威编程模型

目标：

- 让 backend 和 kernel 逐步体现 CG / CPE / LDM / DMA 语义

优先项：

1. attention kernel 数据流
2. KV cache 布局
3. GEMM 的 CG 分块
4. norm / rope 融合

### 阶段 5：趋势级性能分析

目标：

- 只做趋势判断，不做真实性能预测

重点指标：

1. 不同 CG 分块的扩展性
2. attention 与 GEMM 的数据搬运比例
3. KV cache 访问模式
4. prefill / decode 瓶颈切换

## 设计约束

- 继续以 `VLLM-on-SW` 作为主仓库，不再拆出独立的平行实现。
- `sw_emulator` 当前是 correctness/reference backend，不是性能 backend。
- 新能力优先通过 backend 抽象接入，避免直接散落到 model 层和 scheduler 层。
- Python import 暂时保持 `nanovllm`，优先保证兼容性。

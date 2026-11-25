# sapy Phase 0 审计 v1（2026-11-25）

基座：layola13/sla main @ aafd7de 全量复制（sapy 13437f9）。本审计 = 复制树的 crate 级
keep / adapt / drop 清单 + SA 能力对照第一遍。深读项标 [待深读]，不冒充已验证。

## 1. Crate 清单（11 个 workspace 成员，约 130k 行 Rust）

| crate | 行数 | 判定 | 理由 |
|---|---|---|---|
| sla-abi | 575 | KEEP（后改名） | broker/context/descriptor/handler 是宿主 ABI 缝，独立小模块，留作插件形态的余地 |
| sla-lexer | 762 | DROP → 新写 psa-lexer | SLA 词法与 Python 缩进词法不共用；新写 INDENT/DEDENT 词法 |
| sla-ast | 3,554 | DROP → 新写 psa-ast | SLA AST 形状（显式所有权语法）与 .psa AST 不同；arena/span 写法作参考 |
| sla-parser | 8,576 | DROP → 新写 psa-parser | 同上，缩进块 + Python 表达式语法新写 |
| sla-checker | 28,341 | **ADAPT（核心资产）** | rules.rs 与 check_* 模块编码了 SA Referee 语义（借用原点、调用消耗、字段借用、释放事实）——这是推导引擎的知识底座；SLA 表面专属部分（macro/shader/spawn guard/trait 机器）DROP。Phase 2 在此改造：输入从「显式标注」改「推导产出同一组事实」 |
| sla-lowering | 6,109 | ADAPT（保契约） | check_info.rs 是 checker→lowering 的事实契约；psa checker 必须产出同一契约，lowering 本体尽量不动 |
| sla-codegen-text | 34,761 | **KEEP（独立模块）** | SA text 发射层。任务书纪律：保持独立、只经 lowering 输出消费上游类型——sapy 的全部发射都走它，日后与 sla 侧修复双向同步 |
| sla-codegen-sab | 34,465 | DROP（移出 workspace） | SAB 已被用户封存（2026-11-21），sapy 只发 SA text；不测、不修、不带 |
| sla-plugin | 3,835 | DROP（v1） | v1 只做独立 CLI；插件形态后议 |
| sla-oracle | 718 | KEEP | 边界 dump 确定性 oracle，psa 发射确定性检查直接复用 |
| sla-cli | 9,013 | ADAPT → sapy-cli | pipeline.rs / delegate.rs 是「发射 SA 后委托已装 sa 工具链（build-exe/test/build-wasm）」的缝——保留；命令面改 .psa；sab_paths DROP；sa_flattener 保留 |

非 crate 资产：sla_std/（ADAPT：psa 运行时帮手的参考与分叉底）、corpus/ + fixtures/
（KEEP 测试基建形态，psa 语料新写）、scripts/ [待深读]。

依赖方向纪律（继承自基座 Cargo.toml 注释）：单向无环，后端只经 lowering 输出消费
上游类型；HashMap 迭代序绝不进发射路径（确定性由 sla-oracle 把关）。

## 2. SA 能力对照（白皮书 Phase 0 清单，第一遍）

- **发射与后端**：sapy 不写新后端。发射 SA text → 委托已装 `sa` 工具链完成
  build-exe / build-wasm / build-js / build-c / test（sla-cli delegate.rs 已证此缝）。
  Native/WASM 语义一致性由 SA 工具链保证，不由 sapy 假设。
- **Referee**：安全结论最终由 sci 的 SA Referee 对发射出的 SA 判定；Rust checker
  是其规则的镜像（rules.rs）。sapy 的推导只需产出 Referee 可判的指令形态——
  Referee 按实际 load/store 定权限（2026-11-20 裁定）使自动 Borrow 推导可落。
- **Move/Borrow/Release**：checker 的 check_borrow_origin / check_call_consume /
  check_stmt_release / release_facts 已编码现行规则，逐项对照在 Phase 2 做，
  对照不过的语义不许在 psa 层自行发明。
- **容器**：sa_std 的 Vec/Map/String 语义由 sla_std 与 sci .sai 清单定义；
  .psa 的 list/dict/str 映射到这组现有容器，不新造运行时（映射表 Phase 1 出）。
- **动态值**：复制树内**没有**现成的动态值运行时。sla-codegen-text 的
  dyn_coercion.rs 是 SLA 的 dyn trait 强制转换，与 Tagged Value 不是一回事
  [待深读确认]。结论：动态类型（白皮书 Phase 4）是净新工作，立项前不声称支持；
  v1 只做静态可推导子集，动态面一律响亮拒绝并给标注/改写建议。
- **异常/cleanup**：基座以 fallible + 显式控制流表达（SLA 既有形态）；
  .psa 的 try/except 映射留 Phase 3 后，不在 v1。

## 3. 四个未决硬点的落点（对应 TASK.md）

1. 标量 Copy / 容器歧义 → psa-checker 的赋值推导规则（新写，checker 核心改造点）。
2. 动态调用所有权 → v1 无动态调用（见上），规则先行登记：只许借用调用。
3. 元素取出 = 借用、take/pop 显式移出 → psa 容器 API 映射表（Phase 1）。
4. 非拥有引用只许借用域内 → psa-checker 借用域检查复用 check_borrow_origin 的域模型。

## 4. 下一步（骨架改造）

1. workspace 移出 sla-codegen-sab、sla-plugin、sla-lexer、sla-ast、sla-parser
   （先移出构建、文件暂留仓内标 deprecated，一版后删）。
2. 新建 psa-lexer / psa-ast / psa-parser 三 crate（空骨架 + 缩进词法先行）。
3. sla-cli → sapy-cli 改名与 .psa 命令面（build/check 先行，委托缝不动）。
4. 每步 cargo build 绿 + 10 分钟节奏提交。

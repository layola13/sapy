# psa 动态面设计（Tagged Value 运行时）v1 草案（2026-11-25，主会话）

目标：语法完整兼容绝大部分 Python 语法（SYNTAX-CENSUS.md）+ 动态面也
打败 CPython（bench/BASELINE.md 口径）。静态推不断的变量/函数走动态面；
静态面保持零 GC 所有权快路径，两面在同一程序内按变量粒度共存。

## 1. 值表示
- `PsaVal`：显式标签结构体（tag: u8 + payload: u64），不玩 NaN-boxing
  位技巧（与 SA Referee 的指针纪律冲突）。payload：int 存 i64 位、
  float 存 f64 位、bool/none 存小整数、堆对象存指针整数。
- 堆对象统一头：{ rc: u64, kind: u8, ... }，kind = str/list/dict/set/
  tuple/object/function。对象布局用 SLA struct + sa_std 容器实现，
  运行时成模块 `psa_std`（随 sapy 仓维护，与 sla 发射层解耦）。
- int 溢出：v1 先响亮报错（CPython 是任意精度，此为登记在册的兼容
  缺口）；bigint 列 v1 后段，不静默回绕。

## 2. 内存管理（Perceus，动态值限定）
- 引用计数 + 机会性复用（用户 2026-11-25 章程下的主会话工作决定）：
  psa 发射层在已有的所有权追踪点（Ctx 的 moved/borrowed 账本）为 dyn
  值发射 incref/decref；消费型操作（append、setitem、二元运算产新容器）
  在 rc==1 时原地复用、不分配。
- 静态值不进 rc 体系（零 GC 不变）；静态→动态装箱点 incref 一次，
  动态→静态拆箱带运行时类型断言（失败 = 响亮 TypeError 形 panic）。
- 循环引用：v1 不做循环 GC，环会泄漏到进程退出——登记为已知边界
  （CPython 有循环 GC；此项与 bigint 同列 v1 遗留缺口）。

## 3. 静态/动态边界
- 推导单位 = 变量与函数签名：推得出唯一静态类型 → 静态；推不出
  （异构容器、动态调用、eval 形输入、无标注递归多态）→ 整变量 dyn。
- 函数：参数/返回可静态推断则静态签名；否则 dyn 签名
  （PsaVal 进出）。调用点跨界自动装箱/拆箱。
- 动态运算发射为 psa_std 运行时函数调用：dyn_add/dyn_sub/dyn_cmp/
  dyn_index/dyn_call/dyn_getattr 等，按 tag 派发；int/float 快路径在
  运行时函数内先行（双 int 直接 i64 运算），这是动态面赢 CPython 的
  主要来源（编译后的派发 vs 解释器逐字节码派发）。

## 4. 容器与对象
- dyn list = Vec<PsaVal>；dyn dict = sa_std HashMap（键先支持
  str/int/bool，按值哈希）；dyn set 同理；tuple = 定长 Vec<PsaVal>
  + 不可变标记。
- str = 堆字符串对象（字节缓冲 + 长度）；拼接/切片/方法皆运行时函数。
- class：对象 = { class 指针, 属性 dict }；方法查类方法表（dict）；
  单继承链查。静态可知的类后续可降为 SLA struct（优化项，非 v1 门）。
- for 遍历：统一迭代协议函数 dyn_iter_next（list/str/dict/range/
  生成器），生成器用显式状态机结构体（前端把 yield 函数改写为
  状态机，Tier 后段）。

## 5. 异常（最硬项，Tier 1 内最后）
- SA 无异常展开；方案 = 异常槽 + 边界检查：dyn 调用点返回后查
  thread-local 异常槽，非空则沿 try 帧跳转（发射层在每个 dyn 调用后
  插检查，try 体登记 handler）。静态面调用不插检查（零成本）。
- raise/内置异常类型以 dyn 对象表示；except 按类型标签匹配。

## 6. 排期（按 SYNTAX-CENSUS 高频序）
- D0 前端补缺（静态可做，先行）：** 幂、位运算、链式比较、三元、
  f-string、多重赋值脱糖、assert。
- D1 PsaVal + dyn 算术/比较/真值 + 装箱拆箱 + dyn print（对象打印）。
- D2 dyn list/dict/str + 下标/切片 + for 遍历 + in 运算。
- D3 函数动态面：默认参数、关键字参数、*args/**kwargs、lambda/闭包。
- D4 class（属性/方法/单继承）+ 属性访问全集。
- D5 推导式、元组拆包、str/list/dict 方法集高频部分。
- D6 异常（try/except/raise/finally）+ with（文件最小形）+ import
  （自有模块）。
- D7 后段：生成器/yield、async、match、bigint、循环 GC 评估。
- 每期验收：该期语法正例真跑（与 CPython 同源同输出对拍）+ 负例
  响亮 + bench 题集动态写法复测不输 CPython。

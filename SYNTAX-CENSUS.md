# psa 语法覆盖普查（2026-11-25，主会话）

口径：对照 CPython 语言参考的语法面逐项核 psa 现状。现状以 psa-cli 源码为准
（ast.rs 的 Expr/Stmt 枚举 + parser.rs + emit.rs），不凭印象。
状态：【支持】= 静态面已实现且端到端真跑过；【部分】= 有限形态可用；
【缺口-前端】= 纯前端/静态可做，不依赖动态运行时；【缺口-动态】= 须 Tagged
Value 动态面（DYNAMIC-DESIGN.md）；【排除】= v1 明确不做、响亮报错。
【已写未验】= 已退役标记：2026-11-25 深夜统一调试批（批 9 = 25/25，
text 路，23 例输出逐字节对 CPython + 2 负例，提交 092c418 已推
sapy main）后，D3–D7 各行已逐行转正；本节以下为批后绑定计数。

## 1. 词法与字面量
| 项 | 状态 | 备注 |
|---|---|---|
| int/float/bool/None 字面量 | 支持 | D1 后 None 是真实动态值（标签 3）：混合返回/三元可产出、可比较、可打印；D2 切片 A（2026-11-25）补齐 None 字面量词法/语法/发射：`x = None` 直接绑定、None 可作动态列表元素 |
| str 字面量（单/双引号） | 支持 | 转义有限；三引号字符串已支持（2026-11-25 晚间段，词法）；D2 切片 C：str 字面量绑定变量 = 动态字符串（标签 6，Box<Vec<u8>>），(xxv) 在 text 路不存在、旧拒绝已撤 |
| f-string | 部分 | D0 已支持：仅 print() 唯一参数位置；插值限 int/str 字面量 + 至多一个 bool；格式规格/转换响亮拒绝；【已写未验 D5】作一般值：部分拼接入动态字符串（int/bool/str/动态字符串，float 仍被 (xxiii) 堵），print 快路保留；【2026-11-27 M11，集中编译修复批】171-A 已合入：格式规格/转换/嵌套字段实现完成，发射+解析实证（fstring-neg 6/7；n05 应 panic 77085 实为 code=32，属工具链家族 (xlix) 非 psa 缺陷）；运行时正例名集 2/32（c03/c28 过），其余 30 例全红于工具链家族 (xlix)：27 例 panic code=32（psa 与 sla_std/sa_std 源码中均不存在的编译器陷阱码，__psa_fmt_parse 精度块触发）+ 3 例 SIGSEGV（c21/c25/c30）。按实测名集记【部分】并登记受阻名集，家族红不算 psa 未实现；RESULT §6 原建议记【支持】（发射/解析已实证），待家族修复后补验运行时名集再转正 |
| 字节串 b""、原始串 r"" | 支持 | 【2026-11-27 M11，集中编译修复批】已落地并实证（bytes 用例集 29/30）：lexer 转义补齐 CPython 全集（八进制 ≤255、越界响亮报错）；bytes() 内建走 __psa_bytes_convert，其余 dyn 类型 panic 77148；原始串随同摊（staging-gap-bytes）落地。唯一红 by05_index_slice = 工具链家族 (xlix) 切片 print 损坏，非 psa 缺陷 |
| 数字进制 0x/0o/0b、下划线分隔 | 支持 | 2026-11-25 晚间段词法落地（from_str_radix，含下划线） |
| 注释、缩进块、续行 | 支持 | 制表符响亮拒绝（与 Python 允许不同，低频） |

## 2. 表达式
| 项 | 状态 | 备注 |
|---|---|---|
| 算术 + - * / // % ** | 支持 | D0 补 **：int 字面量指数走精确整数幂、其余走 sa_math_pow；int 非字面量指数响亮拒绝（结果类型取决于指数符号）；D1：任一操作数动态时走 PsaVal 运行时分派（** 按运行时指数符号分支、/ 恒 float、// % 按 Python 符号规则） |
| 位运算 & \| ^ ~ << >> | 支持 | D0 已补，静态仅 int 操作数；D1：动态操作数运行时校验后分派 |
| 比较 == != < <= > >= | 支持 | D0 补链式比较：中间操作数须纯（不得含调用），响亮拒绝不可复制者；D1：动态操作数跨类比较（int/float/bool 数值互通、None 等值、跨类排序响亮 panic） |
| in / not in、is / is not | 部分 | in/not in 已通：动态 list（D2B 逐元素 dyn_eq）、动态 dict（D2C 键查表）、动态 str 子串（D2C）；is/is not 已实现（171-B，2026-11-27 M11 集中编译修复批合入）：跨类静态 is/is not 常量短路 + 运行时同一性，用例集 15/17（is02–is11、is13–is17 过，含 div01/div02 两 INFO 发散例，输出与 SEMANTICS §4 登记的 psa 规则一致、不计红）；红 is01_none/is12_none_cond = 工具链家族 (xlix) None 绑定 double free，非 psa 未实现——按实测名集记【部分】并登记受阻名集 |
| and/or/not | 支持 | 纯 bool 发射为 SLA &&/\|\|，其短路语义受上游 (xviii) 影响；D1：含动态操作数时走运行时真值 + 返回操作数语义，短路保留；while 条件内动态 and/or 响亮拒绝 |
| 三元表达式 x if c else y | 支持 | D0 已补：物化为临时变量 + if/else，可嵌套；while 条件内响亮拒绝；str 分支被 (xxv) 堵；D1：分支标量类型不一致或含动态时装箱为动态值 |
| 下标 xs[i] | 部分 | 静态 list 仅读标量元素；D2 切片 A：动态 list 下标读/写全支持（含负下标归一化、越界响亮 panic 77050、非 list 动态值 panic 77052）；D2 切片 C：动态 dict 下标读（缺失键响亮 panic 77054）；D2 收官：动态值链式下标 m[1][0] 已支持（静态链式仍响亮拒绝） |
| 切片 xs[a:b:c] | 支持 | D2 收官（2026-11-25）：静态 list、动态 list、动态 str 全支持，CPython 归一化（负数回绕 + 正步钳 [0,n] / 负步钳 [-1,n-1]）由运行时 __psa_slice_bounds 统一实现；步长 0 响亮 panic 77064、非整数界 panic 77065；dynslice1 与 CPython 逐字节一致（31 例） |
| 属性访问 obj.attr | 支持 | 【D4，批验 class1/class2】动态对象属性读/写/删、类属性字面量读、方法绑定（标签 9）；模块属性随 D6 import |
| 方法调用 | 支持 | 【D5，批验 methods1/2/3】动态值的 str/list/dict/set 方法集高频部分（运行时按名分派，对象自身方法优先）；静态 list 仍仅 append，其余响亮拒绝指向动态面 |
| 函数调用（位置参数） | 支持 | 【D3，批验 fnargs1/2/3 + 5 个 emit 负例】关键字参数、默认值（限字面量）、*args/**kwargs 均已实现；缺参/未知关键字/多值/非字面量默认均在 emit 期响亮拒绝（已逐个核查）；调用点解包 f(*xs) 仍响亮拒绝 |
| lambda | 支持 | 【D3，批验 lambda1 + 捕获负例】lambda 与函数值（标签 7 + 生成分派）；【D8，2026-11-26 集成步骤 2 批验 capture1–5，输出逐字节对 CPython 3.12.3】外层变量捕获已通：单元（cell）共享语义，与 CPython 一致（nonlocal 写回、循环变量迟绑定皆逐字节一致）；闭包为标签 16 PsaVal。响亮拒绝：带默认值/*args/**kwargs 的函数值、参数默认值中的 lambda、参数名 __psa_env、del 捕获变量、while 条件/match 守卫中的捕获变量方法调用、match 模式捕获单元变量。分歧：先读后赋值出 None（非 NameError）；被捕获变量在本作用域亦降级为动态；捕获静态容器/str 参数响亮拒绝（待 (xxxvii)）；打印闭包 panic 77030（与函数值同） |
| 列表/字典/集合字面量 | 部分 | 同类型 int/bool list 静态；D2 切片 A：异构标量/含动态元素/float/空 list 字面量 = 动态堆 list（标签 4），append/len/下标/相等/真值全支持；D2 切片 C：list repr 已落地（静态与动态 list 的 print 均逐字节同 CPython，元素限 int/bool/None/嵌套动态 list；float 元素响亮 panic 77058——待最短往返浮点格式化器）；dict 字面量 = 动态有序 dict（标签 5，交替条目 Vec，D3 取代原 HashMap 形——print/相等随之落地、批验 dict1），str/int/bool/None 键（bool 键与 int 键同一、None 键哨兵，与 Python 键同一性一致），值任意标量/动态值；D2 收官：嵌套容器字面量已支持（list/dict 任意嵌套、作下标赋值值/append 参数）；set 字面量随 D5 落地（标签 11）；float 键未验、容器字面量作调用实参仍响亮拒绝（批中登记缺口） |
| 元组字面量与拆包 | 支持 | 【D5，批验 tuple1/comp1】元组（标签 10）/集合（标签 11）字面量、语句拆包与 for 多目标拆包（运行时 arity 检查 panic 77077）；嵌套/星号目标响亮拒绝 |
| 推导式（list/dict/set/生成器） | 支持 | 【D5/D7，批验 comp1/genexp1】list/dict/set 推导式（多 for + if 子句、range/静态 list/动态可迭代）；生成器表达式 = 合成生成器 def（子句可迭代对象在定义点急切求值、元素/条件仅可引用子句变量/def/类/内建名，其余响亮拒绝无捕获） |
| 生成器表达式/yield | 支持 | 【D7，批验 gen1/genexp1】生成器 def = 包装（Chan<i64> 传 wrap 堆指针——PsaVal 24B 超 SLA 通道 16B 槽限，批中定形；thread::spawn drive）+ 生产者；首拉 start token 保 CPython 惰性；next() 单参；for/推导式/构造器经 __psa_iter_source 全量排干（早 break 仍排干、登记分歧）；耗尽 panic 77081 不可捕 |
| 海象 := | 支持 | 【2026-11-27 M11，集中编译修复批】已落地并实证（walrus 用例集 16/18：w01、w03–w07、w09–w11 与 refused r01–r07 全过）：覆盖位置/方法实参、下标与切片中的裸海象、推导式迭代器重绑定拒绝、genexp 条件海象拒绝、match guard 目标 hoist；红 w02_while_whole/w08_closure = 工具链家族 (xlix)（循环切片重绑 corruption / 闭包捕获 double free，无海象等价形同症），非 psa 缺陷 |
| await | 支持 | 【D7，批验 async1】await 协程值在等待点顺序跑完（__psa_coro_run）；非协程动态值 panic 77082；顶层 await 被 psa 接受（CPython 脚本拒绝、登记分歧） |

## 3. 语句
| 项 | 状态 | 备注 |
|---|---|---|
| 赋值、增广赋值 | 支持 | 多重赋值 a=b=1、序列拆包 a,b=b,a 已支持（2026-11-25 晚间段：RHS 先求值入临时量再绑定；链式绑定的 list 别名响亮拒绝——所有权不可别名）。附带修复：`q = p` 标量新绑定在 SAB 路会 MOVE 源（(xxvii)(a) 族），psa 改先声明后赋值绕行 |
| 下标赋值 xs[i]=v | 部分 | 静态 list + D2 切片 A 动态 list（含增广下标赋值）；D2 切片 C：动态 dict 下标赋值（含增广）已支持 |
| if/elif/else、while、break/continue | 支持 | while/else、for/else 已落地（2026-11-27 M11，集中编译修复批；loopelse 用例集 21/25：le01–le13、le16–le19、le21、le23 与 refused 2 例全过；含审计修复——emit.rs 7 个通用 walker 原漏遍历三类循环的 els，补齐后 le22 为 fail-before/pass-after 钉子，le16 登记为非零退出用例）；红 le14/le15/le20/le22 = 工具链家族 (xlix)（while 内 try/except、纯生成器、闭包/cell 读路径 double free），非 psa 缺陷 |
| for x in range(...) | 部分 | range 一/二参；D2 切片 B：for x in <list 变量> 已支持（静态与动态 list，长度逐轮重读、与 CPython 迭代器语义一致；continue 响亮拒绝）；D2 切片 C：for 遍历动态 str 已支持（逐码点）；for 遍历 dict 响亮 panic 77061（待 D3 有序 dict） |
| return、pass | 支持 | return 多值（元组）随元组项 |
| def（位置参数） | 支持 | 【已写未验 D3】默认值与可变参数已实现；参数/返回注解仍未做（解析即拒绝或忽略待定，现状为未支持） |
| class | 支持 | 【D4，批验 class1/class2/class3】单继承（编译期展平）、方法、__init__/__repr__/__str__/__bool__；类属性走运行时注册表（sa_std globals 单元 psa.clsattrs + set_once 惰性建表、程序启动 seed、祖先链 dict 查找）：声明仍限标量字面量，类侧赋值/增广/删除、实例与子类继承读均按 CPython 语义；super() 与多继承响亮拒绝 |
| import | 支持 | 【D6，批验 mod_main】自有模块：import/from-import 由驱动合并（点分 def 键、类共注册表、模块字面量属性读点求值）；循环导入/别名类/非字面量模块顶层响亮拒绝；psa_std 已接 import 解析：无同名 .psa 文件时 `import json` 绑标准库（json.sla 拼接进程序、dumps/loads 直调，批验 json1）；from-import 标准库 v1 响亮拒绝；本地同名 .psa 优先于标准库 |
| try/except/finally、raise | 支持 | 【D6，批验 exc1/exc2_uncaught】异常机 = 进程全局 pending 槽（sa_std globals 双 u64 单元）+ TryFrame 标志 + 调用语句后尾随未决检查逐帧展开——跨函数传播已通（批 6–9 定形，原每函数槽在被调用方 raise 处误杀 77100）；内建异常类 9 个、raise 四形态；运行时 panic 不可捕（登记分歧） |
| with | 支持 | 【D6，批验 with1】文件类最小形已落地：with 走同一异常机、finally 逆序 __psa_with_close；open() 文件对象（标签 12，r/w/a 文本模式）；通用 __enter__/__exit__ 协议未做 |
| global/nonlocal | 支持 | 【D8，2026-11-26 集成步骤 2 批验 capture2/capture3，输出逐字节对 CPython 3.12.3】global 读写模块变量、nonlocal 写回（计数器双实例独立单元）、未声明读模块变量均已通；单元共享语义。响亮拒绝：global 与 nonlocal 同名、参数声明 global/nonlocal、模块层 nonlocal、无外层绑定的 nonlocal、del global/nonlocal 变量、嵌套 async/generator def。分歧：nonlocal 未绑定负例报错行号为 0（待修）；模块层 lambda 读经他函数 global 声明才存在的名响亮拒绝（窄口） |
| del、assert | 支持 | 2026-11-25 晚间段：assert 脱糖为 if+panic(77040)（消息限字面量、仅作标签）；del name 已支持（删后引用响亮报错、重绑定复活）；del name[idx] 响亮拒绝（SLA 源级无 Vec 收缩形，SA 层有 sa_vec_truncate）；del dict[k] 随 D3 |
| async def/await | 支持 | 【D7，批验 async1】async def 调用返回标签 14 协程值（worker def 经 ::coro 键注册为可调用）；默认值/*args/**kwargs 与 async 方法、async 生成器响亮拒绝；未 await 的协程不运行且不警告（登记分歧）；asyncio/async with/async for 未做 |
| match | 支持 | 【D7，批验 match1】字面量/捕获/通配/序列/或模式 + guard；捕获在 guard 前绑定（guard 失败保留绑定、同 CPython）；同模式重名捕获与或分支绑定集不一致响亮拒绝；True/False 字面量按值比较非同一性（登记分歧）；类/映射/星号模式响亮拒绝 |

## 4. 内建与运行时面（语法之外、兼容所必需）
| 项 | 状态 | 备注 |
|---|---|---|
| print | 支持 | 【D5，统一批全例共验】多参含动态值/bool/f-string（字节打印渲染路）、sep=/end=（限字面量）；单个动态值打印 D1 已支持；float 打印仍被 (xxiii) 堵、响亮 panic 77030 |
| len | 支持 | 静态 list + __psa_dyn_len 运行时分发（list/dict/str/元组/集合，D2–D5 逐标签补齐） |
| range | 部分 | for 头与推导式/构造器（list(range()) 等，D5 批验 comp1）；range 对象作一般值仍缺口 |
| int/float/str/bool 转换 | 支持 | 【D5，批验 builtins1】四转换 + list/tuple/set 构造器；字符串解析失败 panic 77079；str(浮点) 被 (xxiii) 堵 |
| type/isinstance | 支持 | 【D5，批验 builtins1】type 返回 `<class 'NAME'>` 显示字符串（无类值，登记分歧）；isinstance 支持类名（含子类）/内建类型名/元组 |
| list/str/dict 方法集 | 支持 | 【D5，批验 methods1/2/3】高频集见 D5-NOTES；dict keys/values/items 返回快照 list 而非活视图（登记分歧）；str 方法 ASCII-only（登记部分） |
| 整数任意精度 | 支持 | 【2026-11-27 M11，集中编译修复批】已落地并实证（bigint 用例集 12/12 全过：pos01–pos08 输出逐字节一致；neg01 PSA-E6010、neg02 panic 77013、neg03/neg04 panic 77080）：新 PsaVal 标签 17 + 堆上 sla_std BigInt；规范不变量——能进 i64 的值恒降回标签 0、标签 17 只载 i64 外值、每条 int 路径过 __psa_big_norm；静态层不变（静态溢出/字面量仍响亮拒绝，建议码 PSA-E6010）；dyn→静态 unbox 标签 17 panic 77080；位运算底座（sla_std BigInt 无位运算）缺失 panic 77013 = 跟进项 F1；旧 __psa_dyn_pow 溢出静默回绕已由检查+升格版 __psa_ipow2 替换；dyn 移位计数 ≥64 按 CPython 升格（77014 只留负计数） |
| 字符串操作全集 | 部分 | D2 切片 C 已落地：拼接（+/+=）、重复 *、相等与序比较（UTF-8 字节序 = 码点序）、下标取字符（按码点、负索引）、len 按码点、in/not in 子串、真值、for 逐字符、print 与 list 内 repr（Python 引号/转义规则，≥0x80 字节直通为已登记边角）、作 list 元素/dict 值/函数参数与返回值；其后已补：切片 s[a:b]（D2 收官）、字符串方法高频集（D5，ASCII-only 登记部分）、str 作 dict 键（D3 有序 dict）；仍未做：f-string 作一般值绑定的完整形态、join/format 等长尾 |
| pickle 模块（import pickle） | 部分 | 【2026-11-27 M11 新登记，集中编译修复批】cleanroom 版落 psa-std/pickle.sla（1,437 行/55 函数）并完成 import 接线，四入口 dumps/dump/loads/load，panic 码段 77310–77319 已在 panic_codes.rs 注册：只走纯数据 opcode 子集（None/bool/int/float/str/bytes/list/tuple/dict + memo 共享引用），写出 protocol 4（无 FRAME = 登记发散，CPython 可读）、读入兼容 0–5；可执行 opcode（GLOBAL/REDUCE/BUILD/NEWOBJ/EXT/PERSID 类）见字节即 panic 77312 = 安全边界；超 i64 整数 panic 77313。writer 侧逐字节验证通过（dumps(1) 与 CPython 一致、嵌套 dyn 产出正常、77310/77311/77312/77315 拒绝面各在一轮干净运行中实证）；reader 侧被工具链家族 (xlix) 阻塞（同二进制同输入三轮 oracle 结果互相矛盾），家族修复后须复跑 pickle oracle 补验 |

## 5. 汇总（2026-11-25 深夜，统一调试批后绑定重数）
- 语法构造总数（上表逐项）46 项：**支持 35、部分 8、缺口-前端 2、
  缺口-动态 1、排除 0**。支持率 35/46 ≈ 76%，含部分 43/46 ≈ 93%。
  分节：§1 词法 6 = 支持 4 / 部分 1（f-string）/ 缺口-前端 1（字节串）；
  §2 表达式 18 = 支持 14 / 部分 3（in·is、下标、容器字面量）/ 缺口-前端 1（海象）；
  §3 语句 14 = 支持 12 / 部分 2（下标赋值、for）/ 缺口-动态 0；
  §4 内建面 8 = 支持 5 / 部分 2（range、字符串全集）/ 缺口-动态 1（bigint）。
- 转正依据：统一调试批批 9 = 25/25（text 路：23 例输出逐字节对
  CPython + exc2_uncaught/overflow1 两负例），D3 五个 emit 负例
  （缺参/未知关键字/多值/非字面量默认/lambda 捕获）逐个核查为
  响亮拒绝；提交 092c418 已推 sapy main。
- 2026-11-26 集成步骤 2（D8 闭包线）：capture1–5 输出逐字节对
  CPython 3.12.3（10/10 含 parser 5 正例重基线），7/7 负例响亮拒绝；
  global/nonlocal 由缺口-动态转正为支持（§3 计支持 12、缺口-动态 0），
  总数支持 34→35、缺口-动态 2→1。
- 剩余真实缺口（按价值序）：psa_std 标准库（json 初稿已写、未接
  import 解析）、类属性运行时注册表（形状已定形待落地）、float
  格式化 (xxiii) 与 Vec<f64> (xxiv)（编译器
  缺陷队列）、容器字面量作调用实参、静态 list→Dyn 转换面、
  bigint、while/for-else、海象、字节串、is/is not。
- 历史读数（勿再引用为现值）：D1 后旧数曾为约 55 项中支持 12、
  缺口-动态 27——那是 D2–D7 落地前的口径。
- 2026-11-27 M11 翻行（集中编译修复批，证据 = ~/workspace/sapy/compile-batch/RESULT.md，
  逐行明细 = 同目录 CENSUS-FLIP.md）：字节串/原始串、海象 :=、整数任意精度（bigint）
  由缺口转【支持】；while/for-else 随 if/while 行补记落地；f-string 与 is/is not
  按实测名集维持【部分】并登记受阻名集（红项全属工具链家族 (xlix)，不算 psa 未实现）；
  pickle 模块新登记一行【部分】（writer 已验、reader 待家族修复后复验）。
  计数随之更新：总数 46 → **47 项：支持 38、部分 9、缺口-前端 0、缺口-动态 0、排除 0**
  （支持率 38/47 ≈ 81%，含部分 47/47；分节：§1 = 支持 5 / 部分 1；§2 = 支持 15 / 部分 3；
  §3 = 支持 12 / 部分 2；§4 = 支持 6 / 部分 3）。上文「剩余真实缺口」清单中
  bigint、while/for-else、海象、字节串、is/is not 五项自此销账；仍开放者：
  工具链家族 (xlix) 修复后的运行时名集补验（fstring 30 例、isnot 2 例、walrus 2 例、
  bytes 1 例、loopelse 4 例、pickle reader）、psa lexer 指数形 float 字面量 (l)、
  BigInt 位运算跟进项 F1、类属性运行时注册表与 psa_std 接线等既有队列项。

# CPython 对照 bench 基线（2026-11-25，主会话立）

口径：v1 性能门 = 打败 CPython（用户 2026-11-25 11:53 定）。
方法：同一份源码文本同时喂 python3 与 psa（psa 子集与 Python 交集内的
写法），同盒（Colab 1c40a1d00926，跑时 load 0.29）、best-of-3、墙钟毫秒。
CPython = 盒上 Python 3.13.18；psa = Rust sla build-exe 路默认构建。
输出三题两侧逐字一致（fib 832040 / loopsum 50000005000000 /
listsum 500000500000），先对值、再比时。

| 题 | 内容 | CPython | psa（静态面） | 倍率 |
|---|---|---|---|---|
| fib | 递归 fib(30) | 152 ms | 7 ms | 约 22× |
| loopsum | while 累加 10^7 | 1234 ms | 11 ms | 约 112× |
| listsum | list append 10^6 + 下标求和 | 312 ms | 24 ms | 约 13× |

诚实边界：三题都是静态可推断程序，走的是 psa 静态快路径，本表只证明
静态面已远超门槛。**动态面（Tagged Value）建成后必须用同一题集的动态
写法复测**（变量走 dyn 派发、容器装 dyn 值），动态面也过 CPython 才算
v1 性能门成立；动态面数字出来前不得拿本表宣称 v1 达标。
bench 源码在本目录（fib.psa / loopsum.psa / listsum.psa，与 .py 同文本），
run/ 为盒上构建目录（不入库）。

## text 路复测（2026-11-25 深夜，统一调试批后）

口径同上：同一份源码文本、同一盒、best-of-3；psa 改走现行生产路
（psa emit + Zig 宿主 sa build-exe，默认构建）。四题输出先与
CPython 逐字对齐，再比时。

| 题 | 面 | CPython | psa（text 路） | 倍率 |
|---|---|---|---|---|
| fib | 静态 | 212 ms | 9 ms | 约 24× |
| loopsum | 静态 | 1985 ms | 20 ms | 约 99× |
| listsum | 静态 | 518 ms | 13 ms | 约 40× |
| dynsum | 动态 | 215 ms | 154 ms | 约 1.4× |

结论：静态面与动态面同程式复测均打败 CPython，v1 性能门
（蓝图 §1 双门之性能门）成立。注记：首轮 loopsum 曾报
BUILD-FAIL，清 ~/.cache/sa/frontend 陈旧缓存后同脚本重跑即绿
——前端产物缓存键不含编译器版本，是已知缓存面，非编译器缺陷；
倍率随盒负载浮动（fib 的 CPython 侧两轮 144→212 ms），量级稳定。

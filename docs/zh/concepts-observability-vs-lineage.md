# 概念:可观测 vs 血缘

> 一个 Agent 给你一份报告,里面写「中国基本金属铜铝下跌,趋势分 8.5,来源 Sina Finance」。
> 你问它:**这个数是哪来的?能信吗?**
>
> 这一页讲清楚 milkie 用什么回答这个问题——以及它的两种回答**不是一回事**。

milkie 里有两套独立的能力都和「这结论怎么来的」有关,初学时极易混。它们解决的是**不同的信任问题**:

| | 可观测(Observability) | 血缘(Lineage) |
|---|---|---|
| 一句话 | 能回放、读到**「发生了什么」** | 结论带着**指向源的、防伪造的边** |
| 单位 | 一次执行(`run` / `step` / `tool.call`) | 一条 claim → 一个 source object 的 **边** |
| 入口工具 | `get_execution`、Trajectory、`replay` | `get_lineage`、`cite`、`declare_relation` |
| 回答 | 「它跑了 `rhino_report.py`,输出是这坨」 | 「这条结论引用了**这个**源,源不可伪造」 |
| 归因靠谁 | 你/Agent **读那坨输出自己找** | **机械走图**,不读输出 |

两者都建在同一份 event-sourced trace 上(见 [使用指南 §8 Trajectory 与可观测性](./guide.md#8-trajectory-与可观测性)),区别是职责,不是存储。

---

## 1. 可观测:把「发生了什么」钉死

可观测的承诺是:**每次执行都被不可变地记录下来,可回放、可读、可对比。**

- 每次 LLM/工具调用进 append-only 的 event log,带 `causedBy` 因果链。
- `get_execution` 给你一次 run 的**执行投影**:跑了哪些 step、工具的 query、命中的证据、region 怎么组成的。
- 工具的输出本身是**内容寻址**的:跑 `python rhino_report.py`,它的 stdout 被铸成一个 `obj:sha256:…` 对象,改一个字节就是另一个对象。

这已经很强——「它到底跑没跑、跑出了什么」再也赖不掉。**但它回答不了「这一句结论对不对」**:你拿到的是「这次 run 的整坨输出」,要确认某个数,你还得**打开那坨输出、自己在里头找、自己判断**。这「读出来自己找」就是可观测的天花板。

---

## 2. 血缘:把「这结论引用了哪条源」钉成事实

血缘的承诺是:**每条结论携带一条显式、带类型、防伪造的边,指向它的源。**

milkie 的血缘是一张图,两类节点 + 一类边:

- **object(对象)** — 一段被读到或产出的、内容寻址的内容。数据工具(`read_file`/`grep`/shell 等)在取到内容时用 `registerObject` / `citeable` 把它铸成 object,返回一个 `objectId`。类型来自[受控词表](../lineage-taxonomy.md):`passage` / `file` / `claim` / `artifact-blob`,应用可用 `namespace:kind` 扩展(如 `news:item`)。
- **relation(关系)** — 两个 object 之间**带类型、有方向**的边:`cites`(引用)、`derives_from`(派生自)、`supersedes`(取代)、`equivalent_to`(等价)。Agent 用 `cite` / `declare_relation` 工具声明。
- **claim** — Agent 生成的一句结论,本身也是个 object;`cite` 工具把它和源 object 之间连一条 `cites` 边。

查询时,`get_lineage` 折叠 `object.created` / `relation.created` 事件,**顺显式的 `cites` 图走**(注意:走的是声明出来的边,**不是 `causedBy` 因果链**),返回 `claim → sources`。

### 防伪造是结构性的,不是靠提示词

血缘的关键不在「能存边」,在 **`resolveObject` 对不存在的 `objectId` 直接 fail-fast**:

- `cite` / `declare_relation` 拿到一个 Agent 没真领到过的 `objectId` → 直接报错。
- 也就是说,**Agent 没法引用一个它编出来的源**。引用的对象必须是某个数据工具真返回过的。

这就是血缘比可观测**多出来**的那层:归因从「读文本现猜」变成「走一张声明出来的、源不可伪造的图」。

---

## 3. 那份报告,两种回答长什么样

回到开头的问题:报告里「铜铝下跌,趋势分 8.5,来源 Sina Finance」。

- **只有可观测**:Agent 去读自己这次 run 的 `get_execution`,找到跑 `rhino_report.py` 那一步的 stdout,在那坨文本里**搜「铜铝」**,把附近的字念给你。能不能搜到、念得对不对,**全靠这次它读得准不准**——还是 LLM 在文本里做模糊匹配,**仍可能念错、念漏、甚至在文本没覆盖处接着编**。
- **有血缘**:报告里这条信号是个 claim,它带一条 `cites` 边指向某个真实 object。`get_lineage(query="铜铝下跌")` **走图**直接返回那个源 object(`meta` 里就能带 url/source),`resolveObject` 保证这个源是真领到过的、不是编的。归因这一步**没有 LLM**。

---

## 4. 关键陷阱:边是血缘,节点可能还是可观测

这是最容易自我感觉良好、其实没解决问题的地方。

设想最省事的做法:让每条信号 `cite` 到**整坨报告/整次抓取的 stdout** 那**一个** object(shell 工具本来就会把 stdout 铸成 `shell:stdout` object)。这叫**粗粒度 cite**。它**算不算血缘?**

算,但只算一半:

- **那条 `cites` 边,是真血缘**:声明出来的事实、`get_lineage` 走得到、`resolveObject` 防伪造。
- **但它指向的节点,是可观测粒度的**——整坨 stdout 一个 blob。要确认「8.5」「Sina Finance」,你还得**打开这个 blob 自己读**。最后一公里又落回可观测。

> **一张血缘图,精确到什么程度,取决于它的节点有多原子。** 粗粒度 cite = **血缘的边,套在可观测的节点上**。

这正好把「会编」劈成两半,只堵住一半:

| 失败模式 | 粗粒度 cite(边真、节点粗) | |
|---|---|---|
| **源是凭空编的**(根本没这次抓取/没这个 object) | ✅ 结构性堵住(`resolveObject`) | 血缘的功劳 |
| **源是真的,但这个数/这篇在里头没有/不支持** | ❌ 仍可能发生 | 还是可观测:边只绑到「整坨输出」,「数 ∈ 输出」没人机械校验 |

要把第二行也堵掉,只有一条路:**把节点原子化**——每条新闻铸成**一个** `news:item` object(url 进 `meta`),每条信号 `derives_from` 它真正的那几条。这样 claim ↔ 源细到无处张冠李戴,可观测那半也变成了血缘。

---

## 速记

- **可观测**回答「**发生了什么**」;**血缘**回答「**这结论凭哪条源**」。两者都在同一份 trace 上,职责不同。
- 血缘比可观测多的那层,是 **`resolveObject` 让源不可伪造** + **`get_lineage` 走声明出来的图**(不是 `causedBy`)。
- **粗粒度 cite 是「血缘的边 + 可观测的节点」**:能堵「凭空编源」,堵不住「真源里误引」。
- 想两个都堵 → **把节点做细到原子**(每条源一个 object)。粒度,决定血缘到底兑现了多少承诺。

---

延伸阅读:

- [使用指南 §8 — Trajectory 与可观测性](./guide.md#8-trajectory-与可观测性)
- [Lineage Taxonomy — object/relation 受控词表](../lineage-taxonomy.md)
- [设计:#40 lineage / citation 目标](../design/40-lineage-citation-goal.md)
- [如何把数据工具接进血缘](../connecting-data-tools-to-lineage.md)

# ui-focus-proto

给 Agent 做浏览器、手机、桌面 UI 控制时的一套轻量分层原型：**结构化/OCR 快照（`@eN` / `@vN`） + 概率聚焦滤镜（`gemini-decision-1`） + 代际校验点击（`cid`）**。

纯 Python 3.9+ 标准库，零第三方依赖。

---

## 为什么这么拆

常见的 GUI Agent 做法主要有两种极端：
1. **每步截全屏直接喂多模态大模型（VLM）**：延迟高（单步常在 3～8 秒）、Token 消耗大，且让大模型直接报 `(x, y)` 像素坐标容易偏。
2. **直接把整棵 DOM / 无障碍树 / 全屏 OCR 框全塞进上下文**：页面稍复杂就有 150～300 个碎节点，大量无关菜单、状态栏、列表容器刷屏，既浪费上下文又容易干扰模型选错节点。

这套原型的思路是把**“看清有什么”**、**“筛出哪几个跟当前意图有关”**、**“决定下一步干什么”**拆成三件独立的小事：

```text
[浏览器 ARIA 树 / 系统无障碍树] ──▶ @e1..@eN 文本行 ─┐
                                                    ├──▶ [cid=c0001] 纯文本快照 (60~300 行)
[本地轻量 Icon 检测 + OCR 融合]  ──▶ @v1..@vN 文本行 ─┘
                                                             │
                                        ui_focus.py -f "当前这一步想找什么"
                                        (Gemini Decisions API / ~0.8-1.5s)
                                                             │
                                                             ▼
                                    Top-P (0.90) 裁剪后的 5~15 行标星候选
                                    ★ [0.94] [@v6] Vision:icon "Banner Color" center=(720,715)
                                                             │
                                                             ▼
                                    主模型看这几行做决策 -> click @v6 --cid c0001
                                    执行器校验 cid 未过期 -> 查表落精确坐标 (720, 715)
```

三个设计约束：
- **只读聚焦，不越权点按钮**：`ui_focus.py` 只是个类似 `grep` 的语义管道，只负责给 `@eN` / `@vN` 打概率分并裁剪行数，原引用号原样保留，真正怎么操作仍由主模型决定。
- **Fail-Open 兜底**：没配 Key、接口超时、HTTP 报错时，自动原样退回未过滤的完整快照（仅插入一行 `[ui_focus_fallback: ...]`），不会卡断主流程。
- **`cid` 代际防误触**：每次感知快照签发递增的 `cid`（如 `c0001`）。如果模型思考期间页面弹出了新窗口导致状态变化，旧 `cid` 的点击直接拒绝，避免盲点到新界面上。

---

## 目录结构

- `ui_focus.py`：核心语义聚焦滤镜（既可当命令行管道用，也可直接 `import`）。包含 `@ref` 安全键名映射、超 200 节点自动分片并发初筛 + 终审统一尺度、`Top-P`（默认 `0.90`，夹逼 `[5, 15]`）截断与容器噪音剥离。
- `vision_nodes.py`：无障碍树缺失场景（自绘 UI / 游戏 / WebView）的视觉候选归一化参考实现。把上游检测器（如 OmniParser v2）的 `icon` 框与本地 OCR（如 PP-OCRv5）的 `text` 框做 `IoU` 去重与“按钮内文字吸附融合”，输出与 `@eN` 同构的 `@v1..@vN` 文本行，并提供 `SnapshotTracker`（`cid` 校验与坐标查表）。
- `examples/demo_end_to_end.py`：端到端最小串联示例（不配 Key 可离线跑通模拟分布，配了 `GEMINI_API_KEY` 直接走线上真实打分）。
- `tests/test_proto.py`：包含框融合、`cid` 防过期、`Top-P` 过滤、超长分片与 Fail-Open 的单元测试。

---

## 快速跑通

### 1. 跑单元测试与离线演示

```bash
python3 -m unittest discover -s tests -v
python3 examples/demo_end_to_end.py
```

### 2. 配上 Key 跑真实语义聚焦

设置环境变量 `GEMINI_API_KEY`（或写入 `0600` 权限的 `~/.config/ui-focus/gemini-api.key`）：

```bash
export GEMINI_API_KEY="your-api-key"
python3 examples/demo_end_to_end.py
```

### 3. 直接作为 Unix 管道接进你现有的浏览器 / 手机脚本

只要你的现有控制工具能吐出带 `@e1`、`@e2`（或 `@v1`、`@v2`）的文本快照，不用改原工具，直接管道串上去即可：

```bash
# 浏览器快照管道聚焦
your-browser-cli snapshot | python3 ui_focus.py -f "打开账单与发票页面"

# 手机无障碍 + OCR 快照管道聚焦
your-phone-cli perceive | python3 ui_focus.py -f "修改个人资料的横幅颜色"
```

输出效果示意（从几十上百行压缩为几行）：

```text
[cid=c0001]
app=example.settings screen=1440x3120
[ui_focus="change the profile banner color" | kept 5/11 | top1=0.78 | 980ms]
★ [0.78] [@v7] Vision:icon "Banner Color" center=(720,715) bounds=[80,660][1360,770]
★ [0.21] [@v8] Vision:text "#2B2D31" center=(1210,714) bounds=[1120,692][1300,736]
  [0.00] [@v6] Vision:text "Alice" center=(170,528) bounds=[110,506][230,550]
  [0.00] [@v5] Vision:icon label="input_box" center=(720,528) bounds=[80,480][1360,576]
  [0.00] [@v9] Vision:text "Avatar Decoration" center=(250,863) bounds=[80,840][420,886]
```

---

## 边界说明（Non-Goals）

本仓库只提供**核心过滤算子、OCR 框融合逻辑与最小协议原型**，不包含也不计划维护特定平台的底层驱动（如 Playwright 封装、ADB 连接管理、Windows/macOS 无障碍 API、ONNX 推理引擎安装脚本等）。

不同环境的底层截图、OCR 与点击命令差异很大但代码量都很小——建议直接将本仓库的 `ui_focus.py` 与 `vision_nodes.py` 喂给你的 Coding Agent，花几分钟对接到你手头现成的浏览器、手机或桌面控制脚本里即可。

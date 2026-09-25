# 根因分析报告：harness 下 AI 用 bash「复查无失败类事件」容易空跑卡住

> 范围：只做根因定位，不改代码。
> 用户确认的症状：**bash 长时间调用完全不退出**（不是“命令有输出但 AI 反复空转”那种次要形态）。
> 分析对象：DeepSeek Harness（`D:\DSH Desktop\resources\app`，版本 0.1.5-rc.2 / dsh-desktop 0.9.0）。

## 1. 结论摘要

“bash 长时间调用完全不退出”有两条**真正无界**的代码路径，其余路径都有界但会被这两条放大：

1. **进程终止失败后没有任何最终兜底**（最符合“完全不退出”）：
   `bash` 的超时到点只是“尝试”杀掉进程树（SIGTERM → grace → SIGKILL / `taskkill /T /F`）。
   如果杀不掉（Windows 上 `taskkill` 静默失败、权限不足、WSL/孙进程逃逸等），
   底层 `handle.done` 会永远等直接子进程的 `exit` 事件，`bash` 工具调用永不返回，agent 循环永挂。
   同时 `bash` 工具**没有声明定义级 `timeoutMs`**，因此 harness 唯一的全局兜底
   `dsh-tool-call-timeout-policy`（`TOOL_TIMEOUT`）对 `bash` 根本不生效。

2. **后台模式（`run_in_background: true`）明确“无超时”**：
   工具描述写明 “No timeout applies.”，后台进程可以无限运行；
   AI 按提示用 `job_output(wait: true)` 等待时，单次等待有上限（默认 30s、最大 600s），
   但可以无限次重试，整个 turn 表现为“长时间卡住”。

3. **“复查无失败”场景放大问题**：
   `grep`/`ls` 等复查命令经常返回 `(no output)` + exit 0（或 grep 无匹配 exit 1），
   模型得到的信息量极低，倾向于换参数再查；
   而 harness 的重复调用提醒（`dsh-repeat-tool-reminder`）只拦截“参数完全相同的调用”，
   拦截不了“相似但略不同”的空跑复查循环。

---

## 2. 调用链路

```
agent-loop（turn/step 的 while(true)，无每步墙钟）
  └─ dsh-tool-bash.execute()
       ├─ 前台: ctx.shell.run(resolve({...signal}))
       │        └─ dsh-bash-local.LocalBashExecutor.runArgv()
       │             ├─ deadline(signal, timeoutMs)   ← 到点只是 abort
       │             ├─ subprocess.spawn()
       │             │    └─ dsh-subprocess-local
       │             │         ├─ Win32 Job runner（terminateJob，强）
       │             │         └─ fallback（taskkill /T /F，静默失败可能）
       │             └─ await handle.done             ← 直接子进程不退出=永不 resolve
       └─ 后台: jobs.start({run: ...})                ← “No timeout applies.”
                  └─ dsh-jobs-local.wait()             ← 单次有 cap，可无限次
```

---

## 3. 证据链（文件 + 行号）

### 3.1 前台超时只“尝试杀”，没有最终硬截止

- `dsh-bash-local/lib/index.js:128-135`：默认配置 `timeoutMs: 120000`、`maxTimeoutMs: 600000`、`graceMs: 3000`。
- `dsh-bash-local/lib/index.js:164-178`：`resolve()` 把请求超时 clamp 到 `[timeoutMs, maxTimeoutMs]`。
- `dsh-bash-local/lib/index.js:229-256`（`runArgv`）：
  ```js
  const d = deadline(spec.signal, spec.timeoutMs, "BASH_TIMEOUT");
  const handle = this.ctx.subprocess.spawn(...);
  const outcome = await handle.done;   // ← 唯一的等待点
  ```
  超时到点只是触发 `d.signal` abort → 走 kill 流程；**没有任何东西在超时后再给 `handle.done` 一个硬截止**。
- `dsh-subprocess-local/lib/runner-launch-COYGu0Dl.js:894-913`（`directChildResult`）：
  `done` 的唯一事实来源是直接子进程的 `exit` 事件。
- `dsh-subprocess-local/lib/runner-launch-COYGu0Dl.js:1074-1105`（`bindManagedProcess` 的 `done`）：
  `done` 只在 `launch.direct` 完成后 settle（收集模式还会等管道排空，但最多 `graceMs`）。
  直接子进程不退出 ⇒ `done` 永不 settle ⇒ `bash` 工具调用永不返回。
- `dsh-subprocess-local/lib/runner-launch-COYGu0Dl.js:1046-1059`（`terminateWithReason`）：
  SIGTERM → `graceMs` → SIGKILL；SIGKILL 之后**没有后续超时/强制 resolve**。

### 3.2 Windows 兜底 kill 是“静默失败”的

- `dsh-subprocess-local/lib/runner-launch-COYGu0Dl.js:842-852`（`taskkillProcessTree`）：
  ```js
  spawnSync("taskkill", ["/PID", pid, "/T", "/F"], { stdio: "ignore", windowsHide: true });
  ```
  不检查退出码、不看 stderr。权限不足、UAC 提升的子进程、WSL 桥接、任务已逃逸 Job 容器等情况下，
  `taskkill` 可能失败，但 harness 完全不知道。
- `dsh-subprocess-local/lib/runner-launch-COYGu0Dl.js:860-876`（`signalTree`）：
  Windows 分支就是 `taskkill(pid)`。
- `dsh-subprocess-local/lib/runner-launch-COYGu0Dl.js:915-962`（`fallbackOwner`）：
  `alive()` 只看 `child.exitCode === null && child.signalCode === null`；
  kill 失败时该条件一直为真 → `direct` 不 resolve。
- `dsh-subprocess-local/lib/index.js:1025-1026`：代码自己承认弱点——
  “subprocess-local is using weaker process-tree containment because …
  descendants that escape the process group or direct-parent tree are **not guaranteed to terminate or delay waitForExit()**”。
  即没有 Win32 Job runner 时，进程树逃逸即可造成无限等待。
- 有 Win32 Job runner 时（`runner.js` 的 `terminateJob`）更强，但 runner 不可用/加载失败就走上面的 fallback。

### 3.3 `bash` 工具没有定义级 timeoutMs，全局 TOOL_TIMEOUT 兜底缺位

- `dsh-tool-bash/lib/index.js:259-445`（`defineTool({...})`）：
  定义里只有 `name/description/parameters/output/execute/presentCall/presentResult`；
  `timeoutMs` 只在**参数/结果 schema** 中出现（第 273、327 行），**没有定义级 `timeoutMs`**。
- `dsh-tool-call-timeout-policy/lib/index.js:115-141`：
  ```js
  const timeoutMs = ctx.tools.get(exec.name, exec.agent)?.timeoutMs;
  if (timeoutMs === void 0) return next();   // ← bash 命中这里，直接放行
  ```
  即 harness 的 `TOOL_TIMEOUT` 兜底**只对声明了定义级 timeoutMs 的工具生效**，`bash` 不在其中。

### 3.4 后台模式“无超时”，可无限等待

- `dsh-tool-bash/lib/index.js:281-284`：`run_in_background` 参数描述明确写 **“No timeout applies.”**。
- `dsh-tool-bash/lib/index.js:403-427`：后台路径直接 `jobs.start({...})`，没有任何 deadline。
- `dsh-tool-jobs/lib/index.js:22-23`：`waitTimeoutMs` 默认 30000、`maxWaitTimeoutMs` 默认 600000。
- `dsh-tool-jobs/lib/index.js:229-275`：`job_output(wait: true)` 单次最多等 `min(timeout_ms, maxWaitTimeoutMs)`，
  超时返回 `[status: running]` 且 job 继续存活；AI 可以反复调用。
- `dsh-jobs-local/lib/index.js:210-246`：`wait()` 用 `deadline(signal, timeoutMs)` 限时，但只限制“单次等待”，
  不限制 job 本身的存活时间。

### 3.5 agent 循环没有每步墙钟

- `dsh-agent-loop/lib/index.js:638-646`：调度器 `await Promise.race(inFlight.values())`，**没有任何超时**。
- `dsh-agent-loop/lib/index.js:1016`：turn 主体是 `while (true)`。
  ⇒ 任何一个工具 execute 永不 resolve，整个 turn 就永久卡住，直到用户取消或会话 teardown abort。

### 3.6 “复查无失败”为什么特别容易空跑

- `dsh-tool-bash/lib/index.js:54-74`（`renderResult`）：无输出时渲染 `(no output)`；exit 0 时不追加任何 marker。
  模型看到的是一个“成功但没有任何信息”的结果，容易认为证据不足、需要再查一次。
- `dsh-tool-bash/lib/index.js:127`：bash 工具描述鼓励长命令使用 `run_in_background`。
- `dsh-tool-jobs/lib/index.js:204`：系统提示说“不要 busy-poll…… 真正被阻塞时才 `wait: true`”。
  这些提示与“后台无超时”组合，容易把 AI 带进等待一个永不结束的进程。
- `dsh-repeat-tool-reminder/lib/index.js:1385-1389`：提醒只针对 **“完全相同的工具调用 + 完全相同参数”**；
  `dsh-repeat-tool-reminder/lib/index.js:1486` 的 canonical arguments 比对也不覆盖“相似命令 + 连续空输出”。

---

## 4. 根因排序

| 优先级 | 根因 | 表现 | 证据 |
|---|---|---|---|
| P0 | 终止失败后无最终兜底：`handle.done` 永挂 | bash 前台调用“完全不退出”，turn 卡死 | §3.1、§3.2、§3.5 |
| P0 | `bash` 无定义级 `timeoutMs`，`TOOL_TIMEOUT` 全局兜底缺位 | 上面的挂起没有任何第二道保险 | §3.3 |
| P1 | 后台模式“No timeout applies” + 可无限次 `job_output(wait:true)` | 长命令（dev server/tail -f/sleep）永不结束，AI 反复等待 | §3.4、§3.5 |
| P2 | 空输出复查 + 相似重复不拦截 | “确认无失败”类复查循环空转 | §3.6 |

“bash 长时间调用完全不退出”最直接的代码根因是 **P0**：
超时只触发 kill 尝试，kill 失败则 `done` 永不 settle；且 `bash` 工具没有定义级 timeoutMs，
所以 harness 唯一的全局超时策略（`dsh-tool-call-timeout-policy`）也没有把它兜住。
P1 是同一个“卡住”最容易被日常触发的入口（后台长命令本来就不该结束）。

---

## 5. 建议（本次不改代码，仅给后续修复方向）

1. **给 `LocalBashExecutor.runArgv/startArgv` 的 `handle.done` 加最终硬截止**：
   `timeoutMs + graceMs + 固定余量` 后强制 settle，返回 `[timed out …]` / `[killed …]` 结构；
   kill 失败也必须让调用有界返回，而不是永远等进程退出。
2. **让 kill 失败可见**：`taskkillProcessTree` 检查 `spawnSync.status`，失败时走更强手段
   （Win32 Job `terminateJob`、`Stop-Process` 等），并至少把失败记录进结果 marker，而不是静默忽略。
3. **给 `bash` 工具补定义级 `timeoutMs`**（例如 600000），激活 `dsh-tool-call-timeout-policy` 的 `TOOL_TIMEOUT` 兜底，
   使 executor 超时链路失效时仍有第二道保险。
4. **给后台 job 加生命周期上限**（可选 `maxRuntimeMs`/闲置超时），并把“job 本身有界”写进工具描述，
   避免“No timeout applies”成为无限等待的合法理由。
5. **指令层约束复查**：明确“`grep` 无匹配/`(no output)` + exit 0 就是已确认无失败”，
   不要重复同类复查；需要显式证据时用 `grep -q ... && echo FOUND || echo NONE` 一次性给出结果。
6. **增强重复检测**：把“命令相似 + 连续空输出/无进展”也纳入 no-progress 拦截，不限于完全相同参数。

---

## 6. 复现方法（供后续验证修复）

- **前台挂起路径**：`bash` 执行 `sleep 60`，显式 `timeoutMs: 1000`。
  正常应约 1s + 3s grace 内返回 `[timed out after 1000ms]`；
  若调用一直不返回，即命中 P0（kill 失败 + 无硬截止）。
- **后台空转路径**：`bash` 后台执行 `while true; do sleep 1; done`，
  然后 `job_output(wait: true)`。单次会到 cap 返回 `[status: running]`；
  AI 若反复等待/复查，整个 turn 长时间卡住（命中 P1/P2）。

---

## 7. 实证记录（本环境，2026-09-21 会话）

- **前台短超时**：`bash` 执行 `sleep 5`，显式 `timeoutMs: 1000`。
  实际返回：
  ```
  (no output)
  [timed out after 1000ms]
  ```
  说明**普通前台命令的超时 kill 在本机有效**（约 1s 超时 + 3s grace 后返回），
  不是所有 bash 都会挂——挂起只出现在 kill 失败（P0）或后台无超时（P1）路径。

- **后台任务**：`bash` 后台执行 `sleep 120`，返回 job id 后：
  - `job_output(wait: true, timeout_ms: 2000)` 抛错：
    `Error: proc.readOutput is not a function`
  - `job_kill` 抛错：
    `Error: proc.kill is not a function`
  - `job_list` 显示该 job 一直 `running`。

  这是本会话注入层的一个额外观察：**后台 bash job 句柄缺少 `readOutput/kill` 契约**，
  AI 既读不到输出也杀不掉任务，只能等进程自行结束——与“复查/清理时卡住”属于同一类风险，
  并让“无超时后台任务”更容易变成无法回收的悬挂进程。
  （注：这是本环境运行态观察，未展开到源码归属；`dsh-tool-bash` 源码中的后台路径
  期望 job 句柄提供 `readOutput`/`kill`，见 `dsh-tool-bash/lib/index.js:418-425`。）

---

## 8. 涉及文件索引

- `node_modules/@deepseek-ai/dsh-tool-bash/lib/index.js`
- `node_modules/@deepseek-ai/dsh-bash-local/lib/index.js`
- `node_modules/@deepseek-ai/dsh-subprocess-local/lib/index.js`
- `node_modules/@deepseek-ai/dsh-subprocess-local/lib/runner-launch-COYGu0Dl.js`
- `node_modules/@deepseek-ai/dsh-subprocess-local/lib/runner.js`
- `node_modules/@deepseek-ai/dsh-tool-call-timeout-policy/lib/index.js`
- `node_modules/@deepseek-ai/dsh-tool-jobs/lib/index.js`
- `node_modules/@deepseek-ai/dsh-jobs-local/lib/index.js`
- `node_modules/@deepseek-ai/dsh-agent-loop/lib/index.js`
- `node_modules/@deepseek-ai/dsh-repeat-tool-reminder/lib/index.js`

（以上均为本机 DSH 安装目录 `D:\DSH Desktop\resources\app` 下的编译产物，行号以本次读取为准。）

---

## 9. P0 修复记录（2026-09-21，已实施，未改 P1/P2）

本次按“修 P0”范围在 DSH 安装目录直接改了 2 个文件（无源码仓，编译产物即运行代码）：

### 9.1 `node_modules/@deepseek-ai/dsh-bash-local/lib/index.js`

1. **新增配置 `hardKillGraceMs`**（默认 10000ms）：
   - 常量 `DEFAULT_HARD_KILL_GRACE_MS = 1e4`；
   - `LocalBashExecutor.Config` 增加 `hardKillGraceMs: z.number().default(DEFAULT_HARD_KILL_GRACE_MS)`；
   - `assertServiceableBashConfig` 校验正有限且 ≤ `MAX_TIMER_DELAY_MS`。

2. **新增 `settleByDeadline(promise, deadlineMs, onExpiry)`**：
   对任意 Promise 施加绝对墙钟截止；到期先执行 `onExpiry`（终杀尝试），
   再返回 `{ exitCode: null, signal: "SIGKILL" }`，避免调用方永挂。

3. **`runArgv` 前台路径接入硬截止**：
   ```js
   const hardDeadlineMs = spec.timeoutMs + this.config.graceMs + this.config.hardKillGraceMs;
   const outcome = await settleByDeadline(handle.done, hardDeadlineMs, () => {
       try {
           if (typeof handle.terminateForHostExit === "function") handle.terminateForHostExit();
           else handle.terminate();
       } catch (_unterminatable) {}
   });
   ```
   效果：即使 SIGTERM→SIGKILL/`taskkill` 杀不掉进程树、`handle.done` 永不 settle，
   前台 bash 调用也会在 `timeoutMs + graceMs + hardKillGraceMs` 内强制收敛并返回
   `[timed out…]`/`[killed by signal: SIGKILL]`，agent 循环不再无界挂起。

### 9.2 `node_modules/@deepseek-ai/dsh-tool-bash/lib/index.js`

- `defineTool({...})` 增加**定义级** `timeoutMs: 600000`：
  激活 `dsh-tool-call-timeout-policy`（`TOOL_TIMEOUT`）的全局兜底
  （该策略只包裹声明了定义级 `timeoutMs` 的工具，见 `dsh-tool-call-timeout-policy/lib/index.js:115-141`；
  本部署 `timeout-policy` 插件为 active）。

### 9.3 验证

- `node --check` 两个改动文件：`SYNTAX OK`。
- 单元/契约测试 `verify-p0-fix.mjs`（用永不 settle 的假 `handle.done` 直接驱动 `runArgv`）：
  - 挂起路径：**84ms** 内收敛（`hardDeadline=50+10+20`），`timedOut=true`、`signal=SIGKILL`、`exitCode=null`，终杀被调用；
  - 正常路径：`exitCode=0`、`timedOut=false`，无额外终杀，**无回归**。
- 会话内回归：`bash sleep 5` + `timeoutMs:1000` → 仍正常返回 `[timed out after 1000ms]`。
- `dev_reload_package dsh-bash-local` → “缓存中无匹配且磁盘降级失败”：**`dsh-bash-local` 不是 loader entry，
  热重载不适用，harness 重启后生效**；本部署 `tool-bash` 为 [disabled]（由 own-layer shim 注入），
  因此 9.2 的 `timeoutMs` 在启用 loader 插件或源码装配的部署中接管全局兜底，本机前台兜底由 9.1 覆盖。

### 9.4 遗留（不在本次范围）

- P1：`run_in_background: true` 仍“无超时”，后台 job 可无限运行；
- 本会话注入层后台 job 句柄缺少 `readOutput/kill` 契约（§7 实证）；
- P2：空输出复查 + 相似重复检测仍未拦截。

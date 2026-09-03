# Module E: 前端 — 实施计划

> 依赖架构契约：docs/plans/architecture.md §2.1（bench_equity_curve）、§2.4（alive/ic_decay）、§2.9（cooldown 字段 + clear-cooldown）、§2.6（DRL 默认值提示）
> 独占文件：`web/static/index.html`（244KB 单体：HTML ~1960 行 + CSS + JS）；**唯一文件**
> 任务：配合 A/C/D 的 API 变化做前端展示

---

## T1. 回测权益曲线加基准双线（配合模块 A 🔴1）

**Files**：`web/static/index.html`（`renderEquity` :2743-2763、回测结果渲染 :3010/:3027、历史加载 :2555）

**契约**：模块 A 在回测结果 dict 增加 `metrics.benchmark.bench_equity_curve`（数组，长度与 `equity_curve` 一致）。前端在回测权益曲线图叠加一条虚线基准线。

**实现**：
1. `renderEquity(chart, data, key)` 追加第四参数 `benchData=null`（**现有三个参数调用点零改动**，新参数可选）：
   ```js
   function renderEquity(chart, data, key, benchData) {
     nextTick(()=>{
       ...
       const series = [{type:'line', data, showSymbol:false, lineStyle:{color:ct.accent,width:2},
                        areaStyle:{...}}];
       if (benchData && benchData.length === data.length) {
         series.push({type:'line', data:benchData, showSymbol:false,
                      lineStyle:{color:'#94a3b8', width:1.5, type:'dashed'},
                      areaStyle:{opacity:0}});   // 基准线：灰色虚线，无填充
       }
       chart.setOption({..., series});
     });
   }
   ```
2. 回测完成后渲染点（:`3010`、`:3027`）传基准数组：
   ```js
   btResult.value=r; renderEquity(btChart, r.equity_curve.map((v,i)=>[i,v]), 'bt',
     (r.metrics?.benchmark?.bench_equity_curve||[]).map((v,i)=>[i,v]));
   ```
3. 历史回测加载（:2555 `btResult.value` 分支）同样加第四参数。
4. **辅助函数**（模块 E 内加，防多处重复）：
   ```js
   function btBenchData(r){ const b=r?.metrics?.benchmark?.bench_equity_curve; return b?b.map((v,i)=>[i,v]):null; }
   ```

**防回归红线**：
- `renderEquity` 三参调用（:2554、:2608、:3454、:3457、`perfChart` 等 dashboard 曲线）**不传第四参数**，行为与现状逐位一致
- 基准数组长度与 equity_curve 不一致时**不渲染**第二线（防御）：`benchData.length === data.length` 校验
- 图表主题 `ct` 键（accent/accentSoft 等）不变，基准线用固定 `#94a3b8`/灰色（不新增主题键）

**验证**：
- 手工：跑一次回测（demo）→ 权益曲线出现灰色虚线；拖动/缩放/重置正常；dashboard 曲线无变化

---

## T2. 风控冷却"人工确认解除"按钮（配合模块 C 🟠4）

**Files**：`web/static/index.html`（风控面板 :1336-1348、`loadRiskCooldown` :2879-2881、`saveRisk` :2874-2877）

**契约**：模块 C 使 `/api/risk/cooldown` 返回新增 `manual_recovery`/`awaiting_clear` 键，并新增 `POST /api/risk/clear-cooldown` 端点。

**实现**：
1. 冷却状态卡片（:1340-1346）追加"解除"按钮（当 `awaiting_clear` 为 true 时显示）：
   ```html
   <div class="flex items-center justify-between text-[12px]">
     <span class="font-semibold" :class="riskCooldown.active?'neg':'pos'">
       {{riskCooldown.active?(riskCooldown.awaiting_clear?' 冷却已到期 — 等待人工确认':(riskCooldown.manual_recovery?' 冷却中（人工确认模式）':' 冷却中 — 暂停新开仓')):'✅ 冷却状态正常'}}
     </span>
     <span v-if="riskCooldown.active && !riskCooldown.awaiting_clear" class="text-slate-500">剩余 {{Math.ceil(riskCooldown.remaining_sec/60)}} 分钟</span>
     <button v-if="riskCooldown.awaiting_clear" class="mac-btn geen" style="padding:2px 10px;font-size:11px" @click="clearCooldown()">解除冷却</button>
   </div>
   ```
   > 注意：现有 `v-if="riskCooldown.active"` 显示剩余分钟——`awaiting_clear` 时 remaining=0，展示"已到期等待确认"文案覆盖即可，原剩余分钟 span 加条件避免显示 0 分钟。
2. JS 新增：
   ```js
   async function clearCooldown() {
     try { await api('/risk/clear-cooldown', {method:'POST'}); toast('冷却已解除','ok'); loadRiskCooldown(); }
     catch(e){ toast(String(e.message||e),'error'); }
   }
   ```
3. `loadRiskCooldown`（:2879）返回结构自动带上新键（无需改），但初始化 ref（:2878）建议补默认：
   ```js
   const riskCooldown = ref({active:false, remaining_sec:0, consecutive_losses:0, manual_recovery:false, awaiting_clear:false});
   ```
4. 风控规则设置页（:1335-1348 附近）追加"冷却后需人工确认"开关，`saveRisk` 提交时包含该键：
   ```html
   <div class="flex items-center gap-2">
     <label class="mac-label mb-0" for="riskManualRecovery" style="white-space:nowrap">冷却后需人工确认</label>
     <div id="riskManualRecovery" class="mac-toggle" role="switch" :aria-checked="riskRules.risk_manual_recovery?true:false"
          :class="riskRules.risk_manual_recovery?'on':''" @click="riskRules.risk_manual_recovery=riskRules.risk_manual_recovery?0:1"></div>
   </div>
   ```
   `saveRisk`（:2874-2877）PUT `/risk/rules` 的 `riskRules.value` 需含 `risk_manual_recovery`（模块 C 的 `update_rules` 白名单已含该键）。

**防回归红线**：
- 冷却卡片主动展示逻辑：`awaiting_clear` 为 false（默认）时界面与现状一致（只多一个隐藏开关）
- `loadRiskCooldown` 轮询节奏（:3468 每 10s）不变
- 不新增 API；仅消费模块 C 的新端点与新字段

**验证**：
- 手工开启"人工确认"→ 触发冷却 → 到期后面板显示"等待人工确认"+"解除冷却"按钮 → 点击 → 状态恢复正常
- 默认关闭 → 到期自动恢复（现状行为，界面无变化）

---

## T3. DRL 训练面板提示文字（配合模块 D 🟡8）

**Files**：`web/static/index.html`（DRL 训练表单 :1920-1956、`drlCfg` 默认值 :2393、`startDrlTrain` :3277-3320）

**契约**：模块 D 把服务端 `TrainIn` 默认值改为 `entropy_coef=0.05`、`reward_trend_align=0.1`、`reward_dd_penalty=0.5`、`reward_losing_penalty=0.2`。前端在训练面板加推荐提示。

**实现**：
1. `drlCfg`（:2393）默认值已含 `entropy_coef:0.05`（与 D 一致），无需改。**新增**三个塑形参数字段（供用户覆盖，默认对齐 D 推荐值）：
   ```js
   const drlCfg = reactive({..., entropy_coef:0.05, reward_dd_penalty:0.5, reward_losing_penalty:0.2, reward_trend_align:0.1, ...});
   ```
2. `startDrlTrain` 的请求体（:3281-3286）追加：
   ```js
   ..., entropy_coef:drlCfg.entropy_coef,
   reward_dd_penalty:drlCfg.reward_dd_penalty, reward_losing_penalty:drlCfg.reward_losing_penalty,
   reward_trend_align:drlCfg.reward_trend_align, ...
   ```
3. 训练面板（:1957 附近、训练按钮上方）加提示文字：
   ```html
   <div class="text-[11px] text-slate-500 mb-2">推荐默认值：entropy_coef=0.05, trend_align=0.1, dd_penalty=0.5, losing_penalty=0.2（可在下发参数中覆盖）</div>
   ```

**防回归红线**：
- 不传新字段时（前端未改）→ 后端默认值生效，行为仍正确（新旧前端兼容）
- `drlTrain` 轮询进度结构不变；提示文字为纯展示
- 输入项不阻塞：塑形参数在 UI 可不暴露输入（仅提示文字），默认走后端值即可——**最小实现只做提示文字**

**验证**：
- 手工：DRL 页出现提示文字；`POST /api/drl/train` 日志含 `[drl] 奖励塑形已启用: dd=0.5 losing=0.2 trend=0.1`

---

## T4. 因子库页面扩展（配合模块 B 🔴2 / 🟠6 / 🟡10）

**Files**：`web/static/index.html`（因子库 pills :1687-1693、`loadFactorLibrary` :3156-3157、`mineFactors` :3252-3265）

**契约**：模块 B 使 `/api/factors/library` 的 `Factor.meta()` 返回追加 `alive`（bool）与 `ic_decay`（dict）键；`/api/factors/mine` 返回各候选带 `cross_validation`（passed/failed/not_configured）键；`/factors/mine` 接受 `cross_symbols`。

**实现**：
1. 因子库 pills（:1689-1691）加下线状态标记：
   ```js
   // pill 内追加状态徽标
   {{f.name}}<span v-if="f.alive===false" class="ml-1 text-[10px]">{{' ❌'}}</span>
   ```
   以及 title 附加状态（:1689 `:title="f.description"` → 动态拼 alive/ic_decay 摘要）：
   ```js
   :title="f.description + (f.alive===false?'（已下线：IC 衰变）':f.ic_decay?.auto_offline?'（曾下线后恢复）':'')"
   ```
   > 说明：`v-if="f.alive===false"` 使用严格判断——旧缓存/旧数据无 alive 键时按活跃处理（与模块 B 默认 `alive=True` 一致），不渲染 ❌。
2. `mineFactors`（:3252-3265）：改为发送 `cross_symbols`（从新增输入控件取值）+ 展示 `cross_validation` 徽标：
   - 挖掘面板（:1798-1801）加跨品种输入：
     ```html
     <div class="flex gap-2 items-center mt-2">
       <span class="text-[11px] text-slate-500">跨品种验证(逗号分隔，exchange源): </span>
       <input v-model="factorCrossSymbols" class="mac-input" style="width:220px" placeholder="ETH/USDT,BNB/USDT"/>
     </div>
     ```
   - 请求体（:3255-3257）加 `cross_symbols:factorCrossSymbols.value`；`minedFactors` 结果卡片（:1809-1817）追加：
     ```js
     <span class="pill" :class="f.cross_validation==='passed'?'green':(f.cross_validation==='failed'?'red':'slate')">
       {{f.cross_validation==='passed'?'✓跨品种通过':f.cross_validation==='failed'?'✗跨品种未通过':'⚠未配置'}}
     </span>
     ```
3. 新增 ref：`const factorCrossSymbols = ref('');`（:2384 附近）并将其加入返回（:3507-3510 setup 返回列表）。

**防回归红线**：
- 无 alive 键的旧因子数据不报错、不渲染下线徽标（严格 `===false` 判断）
- `momentum_1w`/`vol_spike_20`/`price_pos_20` 等新因子出现为纯增量（模块 B 交付后）
- cross_symbols 输入为空时请求体照旧（不带字段）→ 后端 not_configured，前端显示 ⚠，不阻断挖掘

**验证**：
- 手工：`/api/factors/library` 有 alive 键 → 下线因子在 pills 上显示 ❌；`POST /factors/mine`（exchange+ETH/USDT）→ 结果卡片显示跨品种徽标
- 回归：demo 源挖掘（无 cross_symbols）→ ⚠ 未配置，结果照常显示

---

## 交付验收清单（模块 E）

- [ ] `node scripts/check_js.js`（或项目内 JS 语法检查脚本）通过——index.html 内联 JS 无语法错
- [ ] 手工启动 `python run.py web` 打开页面：4 个 Pane 无白屏、控制台无 JS 错误
- [ ] 回测权益曲线双线（模块 A 合入后联调）
- [ ] 风控冷却解除按钮 + 人工确认开关（模块 C 合入后联调）
- [ ] DRL 面板提示文字出现
- [ ] 因子库/挖掘跨品种徽标 + 下线标记（模块 B 合入后联调）
- [ ] 现有功能回归：加载历史回测、性能曲线、dashboard equity 曲线等无变化
- [ ] 记录模块 A/C/D API 字段与前端消费的对齐情况（联调记录）
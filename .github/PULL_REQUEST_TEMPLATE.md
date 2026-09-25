## 变更说明

<!-- 一句话说明本次改了什么、为什么。 -->

## 涉及范围

- [ ] 依赖管理（requirements* / lock / .in）
- [ ] CI / 工具链（.github、scripts、pyproject、ruff、coverage）
- [ ] 核心运行时（core / engine / exchange / backtest / strategies / factors / drl / ai / web）
- [ ] 前端（web/static/index.html）
- [ ] 文档（docs / README）

## 验证

提交前请跑通对应门禁（docs/REDEV_GUIDE.md §4）：

- [ ] `compileall` 通过
- [ ] `ruff check`（E9+F+W+E4+E7）通过
- [ ] `mypy` 试点（若涉及试点文件）通过
- [ ] `check_js.py` / `build_frontend.py --check-only` 通过（若涉及前端）
- [ ] 全量 / 快路径 `pytest` 通过
- [ ] 期刊：显示覆盖/基线是否变化

## 红线检查

- [ ] 已阅读 `docs/REDEV_GUIDE.md` §3「禁止回退清单」，确认未破坏任何红线
- [ ] 有意变更基线的，已更新 §1 基线表或 §6 记录

## 附注

（向后兼容性 / 回滚方法 / 待补测试 等）
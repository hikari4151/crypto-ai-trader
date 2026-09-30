#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 echarts 按需打包产物（web/static/vendor/echarts.custom.min.js）。

背景（第十三轮性能优化，2026-09-29）：
- 原 vendor 为 echarts 官方完整包 echarts.min.js（1005.1KB / gzip 326.7KB），
  是全站最大静态资源、前端"启动到可用"的主导项。
- index.html 实际只用了 6 类图表（line/bar/scatter/candlestick/heatmap/tree）
  + grid/tooltip/legend/title/dataZoom/visualMap/markLine/axisPointer + canvas 渲染。
- 按需打包产物 636.7KB / gzip 211.0KB：体积 -36.6%、gzip 传输 -35.4%，
  echarts 全局 API 面（echarts.init 等）与完整版一致，渲染行为等价（见下文验证）。

用法：
  python scripts/build_echarts_custom.py            # 需要网络（npm i echarts@5.5.0 esbuild）
  产物：web/static/vendor/echarts.custom.min.js + 打印 SRI sha384

注意：
- 生成后必须同步更新 web/static/index.html 中该 script 标签的 integrity
  （SRI 不匹配浏览器会拒载脚本）。
- vendor 走 immutable 长缓存：内容变更须改文件名（echarts.custom.min.js），
  旧文件名 echarts.min.js 已删除，勿复用旧名。
- 完整性验证：本包与完整包在真实页面的渲染行为等价——
  (1) .optim/echarts_custom/smoke.html 六类图表全部 DONE_PAINT_err0；
  (2) ?view= 逐视图 A/B：dashboard/strategy/evolve 两包均稳定绘制 1/1，
      performance 的 0/0 波动两包同现（API 数据时序，非包差异），
      backtest/quant 无容器是数据门控（btResult/factorIcs 为空）两包一致。
"""
import hashlib
import base64
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "web" / "static" / "vendor"
OUT = VENDOR / "echarts.custom.min.js"

ENTRY = r"""
import * as echarts from 'echarts/core';
import {
  LineChart, BarChart, ScatterChart, CandlestickChart, HeatmapChart, TreeChart,
} from 'echarts/charts';
import {
  GridComponent, TooltipComponent, LegendComponent, TitleComponent,
  DataZoomComponent, VisualMapComponent, MarkLineComponent, AxisPointerComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

echarts.use([
  LineChart, BarChart, ScatterChart, CandlestickChart, HeatmapChart, TreeChart,
  GridComponent, TooltipComponent, LegendComponent, TitleComponent,
  DataZoomComponent, VisualMapComponent, MarkLineComponent, AxisPointerComponent,
  CanvasRenderer,
]);

// 与官方完整版全局面保持一致（完整版把 echarts 挂到 globalThis.echarts）。
// ⚠️ 不能 export 也不能传 --global-name：esbuild 的 `var echarts=` 顶层包装
// 会把 {default} 命名空间盖过显式赋值。裸 IIFE + 副作用赋值才是正确形态。
globalThis.echarts = echarts;
"""


def sh(cmd: list[str], cwd: Path) -> None:
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        sys.exit(f"命令失败 {cmd!r}\n{r.stdout}\n{r.stderr}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-only", action="store_true",
                    help="只打印入口模块清单，不执行构建（离线检查用）")
    args = ap.parse_args()

    if args.entry_only:
        print(ENTRY.strip())
        return

    with tempfile.TemporaryDirectory(prefix="echarts-build-") as td:
        work = Path(td)
        sh(["npm", "init", "-y"], work)
        sh(["npm", "install", "echarts@5.5.0", "esbuild"], work)
        (work / "entry.js").write_text(ENTRY, encoding="utf-8")
        sh(["npx", "esbuild", "entry.js", "--bundle", "--minify",
            "--format=iife", "--outfile=echarts.custom.min.js"], work)
        data = (work / "echarts.custom.min.js").read_bytes()

    OUT.write_bytes(data)
    b64 = base64.b64encode(hashlib.sha384(data).digest()).decode()
    print(f"产物：{OUT.relative_to(ROOT)}  {len(data)} bytes")
    print(f"SRI ：sha384-{b64}")
    print("请更新 web/static/index.html 对应 script 标签的 integrity 为该 SRI。")


if __name__ == "__main__":
    main()

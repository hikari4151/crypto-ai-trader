"""验证 AI 守卫前端 UI：程序方向对比/连续性提示/交易者方程区块渲染，无 JS 错误。"""
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

DRIVER = r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\geckodriver\geckodriver.exe"
BASE = "http://127.0.0.1:8000"


def main():
    opts = Options()
    opts.add_argument("--headless")
    opts.set_preference("intl.accept_languages", "zh-CN")
    svc = Service(DRIVER)
    drv = webdriver.Firefox(service=svc, options=opts)
    try:
        drv.execute_script("window.__e=[];const _o=console.error;console.error=function(...a){window.__e.push(a.map(String).join(' '));_o.apply(console,a);}")
        drv.set_window_size(1400, 900)
        drv.get(f"{BASE}/?view=dashboard")
        time.sleep(3.5)
        # 检查新 UI 区块是否出现在页面源（即使无分析结果，模板也应存在）
        html = drv.page_source
        probes = ["程序结构复算", "决策连续性", "交易方案"]
        for p in probes:
            print(f"[probe] '{p}' -> {'YES' if p in html else 'NO'}")
        logs = drv.execute_script("return window.__e || []")
        print(f"console errors: {len(logs)}")
        for l in logs[:6]:
            print("  ERR:", l[:200])
        drv.save_screenshot(r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\shot_ai_guards.png")
        print("saved shot_ai_guards.png")
    finally:
        drv.quit()


if __name__ == "__main__":
    main()

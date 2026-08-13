"""验证前端多视图无 JS 错误 + 策略参数 choices 下拉渲染。"""
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
        for view in ["dashboard", "backtest", "quant", "strategy", "settings", "market", "performance"]:
            drv.execute_script("window.__e=[];const _o=console.error;console.error=function(...a){window.__e.push(a.map(String).join(' '));_o.apply(console,a);}")
            drv.get(f"{BASE}/?view={view}")
            time.sleep(2.5)
            logs = drv.execute_script("return window.__e || []")
            print(f"[{view}] console errors: {len(logs)}")
            for l in logs[:4]:
                print("   ERR:", l[:160])
        # 切到 strategy 视图检查参数下拉（factor_signal 有 choices）
        drv.get(f"{BASE}/?view=strategy")
        time.sleep(2.5)
        info = drv.execute_script("""
          const selects = document.querySelectorAll('section select');
          const hasParamSel = [...document.querySelectorAll('section .mac-card')].some(c =>
            c.textContent && c.textContent.includes('当前策略参数') && c.querySelector('select'));
          return {select_count: selects.length, param_panel_has_select: hasParamSel};
        """)
        print("策略参数面板检查:", info)
    finally:
        drv.quit()


if __name__ == "__main__":
    main()

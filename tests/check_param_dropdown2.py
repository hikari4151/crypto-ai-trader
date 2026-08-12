"""通过 API 切换 factor_signal 后，检查参数面板 choices 下拉。"""
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
        # 先通过 API 把引擎切换到 factor_signal
        import urllib.request
        req = urllib.request.Request(f"{BASE}/api/trading/strategies/select",
                                     data=b'{"name":"factor_signal"}',
                                     headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req).read()
        drv.get(f"{BASE}/?view=strategy")
        time.sleep(3)
        info = drv.execute_script("""
          const wins = [...document.querySelectorAll('section .mac-window')];
          const panel = wins.find(w => w.textContent && w.textContent.includes('当前策略参数'));
          if (!panel) return {found:false};
          const selects = [...panel.querySelectorAll('select')].map(s => {
            const label = s.closest('div')?.querySelector('label')?.textContent || '';
            return {label, options:[...s.options].map(o=>o.value)};
          });
          return {found:true, selects};
        """)
        print("参数面板:", info)
        logs = drv.execute_script("return window.__e || []")
        print("console errors:", len(logs))
        ok = info.get("found") and any(s.get("label") == "因子" and len(s.get("options", [])) >= 2 for s in info.get("selects", []))
        print("RESULT:", "PASS" if ok else "FAIL")
    finally:
        drv.quit()


if __name__ == "__main__":
    main()

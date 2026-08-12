"""验证策略参数 choices→下拉：切换到 factor_signal 后参数面板应显示 select。"""
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import Select

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
        drv.get(f"{BASE}/?view=strategy")
        time.sleep(3)
        # 找到"选择策略"下拉并切到 factor_signal
        sel = Select(drv.find_element(By.XPATH, "//select[option[text()='factor_signal']]"))
        sel.select_by_value("factor_signal")
        time.sleep(1.5)
        # 检查参数面板下拉（面板是 .mac-window，标题"当前策略参数"）
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
        print("参数面板下拉:", info)
        logs = drv.execute_script("return window.__e || []")
        print("console errors:", len(logs))
        for l in logs[:4]:
            print("  ERR:", l[:160])
        # 应有 factor 与 mode 两个多选项下拉
        hasDropdown = info.get("found") and any(len(s.get("options", [])) >= 2 for s in info.get("selects", []))
        print("RESULT:", "PASS" if hasDropdown else "FAIL")
    finally:
        drv.quit()


if __name__ == "__main__":
    main()

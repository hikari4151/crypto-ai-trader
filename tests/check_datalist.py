"""验证 datalist 主流币种选项是否生效。"""
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

DRIVER = r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\geckodriver\geckodriver.exe"


def main():
    opts = Options()
    opts.add_argument("--headless")
    opts.set_preference("intl.accept_languages", "zh-CN")
    svc = Service(DRIVER)
    drv = webdriver.Firefox(service=svc, options=opts)
    try:
        drv.execute_script("window.__e=[];const _o=console.error;console.error=function(...a){window.__e.push(a.map(String).join(' '));_o.apply(console,a);}")
        drv.set_window_size(1400, 900)
        drv.get("http://127.0.0.1:8000/?view=backtest")
        time.sleep(3)
        info = drv.execute_script("""
          const dl = document.getElementById('popularSymbolList');
          const inp = document.querySelector('input[list="popularSymbolList"]');
          return {datalist_options: dl ? dl.querySelectorAll('option').length : -1,
                  backtest_input: inp ? inp.getAttribute('placeholder') : 'NOT FOUND',
                  list_attr: inp ? inp.getAttribute('list') : null};
        """)
        print("backtest视图:", info)
        # 切到 settings 视图检查 AI 模型 datalist
        drv.get("http://127.0.0.1:8000/?view=settings")
        time.sleep(2.5)
        ai_info = drv.execute_script("""
          const aidl = document.getElementById('aiModelList');
          const aiInp = document.querySelector('input[list="aiModelList"]');
          return {ai_datalist_options: aidl ? aidl.querySelectorAll('option').length : -1,
                  ai_model_input: aiInp ? 'linked' : 'NOT FOUND',
                  provider: (document.querySelector('select') ? document.querySelector('select').value : '?')};
        """)
        print("settings视图 AI模型:", ai_info)
        logs = drv.execute_script("return window.__e || []")
        print("console errors:", len(logs))
        for l in logs[:5]:
            print("  ERR:", l[:180])
    finally:
        drv.quit()


if __name__ == "__main__":
    main()

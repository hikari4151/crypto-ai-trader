"""端到端验证跨行情泛化评估面板：选模型→点评估→结果渲染。"""
import sys
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service

BASE = "http://127.0.0.1:8000"


def run() -> int:
    opts = Options()
    opts.add_argument("--headless")
    opts.set_preference("intl.accept_languages", "zh-CN")
    opts.set_preference("dom.webnotifications.enabled", False)
    svc = Service(r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\geckodriver\geckodriver.exe")
    drv = webdriver.Firefox(service=svc, options=opts)
    try:
        drv.execute_script("window.__e=[];const _o=console.error;console.error=function(...a){window.__e.push(a.map(String).join(' '));_o.apply(console,a);}")
        drv.set_window_size(1400, 900)
        drv.get(f"{BASE}/?view=quant")
        time.sleep(3.0)
        # 滚动到跨行情面板
        drv.execute_script("""
          const els = [...document.querySelectorAll('section *')].filter(e=>
            e.textContent && e.textContent.includes('跨行情泛化评估'));
          if (els[0]) els[0].scrollIntoView({block:'center'});
        """)
        time.sleep(0.8)
        # 面板内的 select 顺序：评估模型 / 数据源 / 周期
        from selenium.webdriver.support.ui import Select
        panel = drv.execute_script("""
          const els = [...document.querySelectorAll('section *')].filter(e=>
            e.textContent && e.textContent.includes('跨行情泛化评估'));
          els[0].scrollIntoView({block:'center'});
          return els[0] ? true : false;
        """)
        time.sleep(0.5)
        sels = drv.find_elements(By.CSS_SELECTOR, "section .mac-card select")
        # 定位跨行情面板（最后一个 mac-card 含跨行情文本）——直接用全部 select 过滤父容器
        all_sels = drv.execute_script("""
          const panel = [...document.querySelectorAll('section *')].filter(e=>
            e.textContent && e.textContent.includes('跨行情泛化评估'))[0];
          return [...panel.querySelectorAll('select')].map(s=>({label: s.previousElementSibling?.textContent||'', options:[...s.options].map(o=>o.value)}));
        """)
        print("[e2e-eval] 面板下拉框:", all_sels)
        # 模型下拉框：label 为"评估模型"
        model_sel = Select(drv.find_element(By.XPATH,
            "//*[text()='评估模型']/following-sibling::select[1]"))
        opts = model_sel.options
        if len(opts) > 1:
            model_sel.select_by_index(len(opts) - 1)  # 选最新模型
        print("[e2e-eval] 已选择模型:", model_sel.first_selected_option.text if len(opts) > 1 else "无")
        # 点击跨行情面板内的评估按钮（面板内唯一 primary 按钮，文本含 🔬）
        drv.execute_script("""
          const panel = [...document.querySelectorAll('section *')].filter(e=>
            e.textContent && e.textContent.includes('跨行情泛化评估'))[0];
          const btn = [...panel.querySelectorAll('button')].find(b =>
            b.textContent.includes('评估') && b.textContent.includes('🔬'));
          if (btn) btn.click();
        """)
        print("[e2e-eval] 已点击跨行情面板评估按钮")
        # 等待结果出现
        deadline = time.time() + 40
        found = False
        txt = ""
        while time.time() < deadline:
            time.sleep(1.5)
            txt = drv.execute_script("""
              const panel = [...document.querySelectorAll('section *')].filter(e=>
                e.textContent && e.textContent.includes('跨行情泛化评估'))[0];
              return panel ? panel.innerText : '';
            """)
            if '总收益' in txt and ('泛化' in txt or '评估' in txt) and ('评价' not in txt):
                found = True; break
        if not found:
            print("[e2e-eval] 超时未等到结果，面板文本:", txt[:400])
            drv.save_screenshot(r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\e2e_oos_eval_timeout.png")
            return 2
        drv.execute_script("""
          const els = [...document.querySelectorAll('section *')].filter(e=>
            e.textContent && e.textContent.includes('跨行情泛化评估'));
          if (els[0]) els[0].scrollIntoView({block:'center'});
        """)
        time.sleep(0.6)
        drv.save_screenshot(r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\e2e_oos_eval_done.png")
        print("[e2e-eval] 面板文本:")
        print(txt[:600])
        logs = drv.execute_script("return window.__e || []")
        print(f"[e2e-eval] console errors: {len(logs)}")
        for l in logs[:8]:
            print("  ERR:", l[:200])
        ok = '总收益' in txt and '夏普' in txt
        print(f"[e2e-eval] RESULT: {'PASS' if ok else 'CHECK'}")
        return 0 if ok else 3
    except Exception as e:  # noqa: BLE001
        print(f"[e2e-eval] 失败: {e}")
        return 4
    finally:
        drv.quit()


if __name__ == "__main__":
    sys.exit(run())

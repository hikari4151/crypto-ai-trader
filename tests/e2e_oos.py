"""端到端验证：触发真实 DRL 训练 → 训练完成卡片渲染 OOS 独立评估报告。

步骤：
1. 加载 quant 视图
2. 把训练轮数改为 30、K线改为 900（加速）
3. 点击"开始训练"
4. 轮询等待"训练完成"卡片出现
5. 检查卡片中的 OOS 报告内容与 JS 错误
"""
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
        # 加速：训练轮数 30
        drv.execute_script("""
          const inp = document.querySelector('input[type="number"][min="10"]');
          if (inp) { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set;
                     setter.call(inp,'30'); inp.dispatchEvent(new Event('input',{bubbles:true})); }
        """)
        time.sleep(0.5)
        # 点击开始训练
        btns = drv.find_elements(By.CSS_SELECTOR, "button")
        clicked = False
        for b in btns:
            if '开始训练' in (b.text or ''):
                b.click(); clicked = True; break
        if not clicked:
            print("[e2e] 未找到开始训练按钮"); return 1
        print("[e2e] 已点击开始训练")
        # 轮询等待训练完成卡片
        deadline = time.time() + 180
        found = False
        while time.time() < deadline:
            time.sleep(3)
            try:
                card = drv.find_element(By.XPATH, "//*[contains(text(),'样本外(OOS)独立评估')]")
                if card.is_displayed():
                    found = True; break
            except Exception:
                pass
            try:
                err = drv.find_element(By.XPATH, "//*[contains(text(),'训练失败')]")
                if err.is_displayed():
                    print("[e2e] 训练失败"); break
            except Exception:
                pass
        if not found:
            print("[e2e] 超时未等到 OOS 卡片")
            drv.save_screenshot(r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\e2e_oos_timeout.png")
            return 2
        # 截图训练完成卡片
        try:
            drv.execute_script("""
              const els = [...document.querySelectorAll('section *')].filter(e=>
                e.textContent && e.textContent.includes('样本外(OOS)独立评估'));
              if (els[0]) els[0].scrollIntoView({block:'center'});
            """)
            time.sleep(0.8)
            drv.save_screenshot(r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\e2e_oos_done.png")
        except Exception as e:
            print("[e2e] 截图失败:", e)
        # 读取卡片文本内容
        txt = drv.execute_script("""
          const card = [...document.querySelectorAll('section *')].filter(e=>
            e.textContent && e.textContent.includes('样本外(OOS)独立评估'))[0];
          return card ? card.innerText.slice(0,500) : 'NO CARD';
        """)
        print("[e2e] OOS 卡片内容:")
        print(txt)
        logs = drv.execute_script("return window.__e || []")
        print(f"[e2e] console errors: {len(logs)}")
        for l in logs[:8]:
            print("  ERR:", l[:200])
        ok = 'OOS收益' in txt and ('疑似过拟合' in txt or '泛化正常' in txt)
        print(f"[e2e] RESULT: {'PASS' if ok else 'CHECK'}")
        return 0 if ok else 3
    except Exception as e:  # noqa: BLE001
        print(f"[e2e] 失败: {e}")
        return 4
    finally:
        drv.quit()


if __name__ == "__main__":
    sys.exit(run())

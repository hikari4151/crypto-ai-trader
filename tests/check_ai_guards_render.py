"""验证 AI 守卫前端区块真实渲染：注入模拟 analysis 数据后截图。"""
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
        time.sleep(3.0)
        # 注入模拟 analysis 数据（含 _program 方向冲突 + _continuity 反手 + trade_plan）
        drv.execute_script("""
          const vm = document.querySelector('#app').__vue_app__;
          // Vue3 setup 暴露的 reactive 状态通过组件实例访问比较麻烦，
          // 直接尝试通过 DOM 事件触发按钮不可靠，改为访问 app 内部
          // 这里用更稳妥方式：通过 Vue 组件的 provide/inject 不可行，
          // 直接尝试把 analysis 挂到 window 再由事件赋值。
        """)
        # 备用：直接用 JS 触发按钮点击走真实 analyze（更真实）
        btns = drv.find_elements("css selector", "button")
        target = None
        for b in btns:
            if "立即解读" in (b.text or ""):
                target = b
                break
        if target:
            print("找到『立即解读』按钮，点击触发真实分析…")
            target.click()
            # 等待分析完成（可能数十秒，轮询）
            deadline = time.time() + 120
            done = False
            while time.time() < deadline:
                time.sleep(3)
                # 检查解读内容是否出现（summary 非空）
                has = drv.execute_script("""
                  const cards = [...document.querySelectorAll('section .mac-card, section .mac-window')];
                  return cards.some(c => c.textContent && c.textContent.includes('市场:'));
                """)
                if has:
                    done = True
                    break
            print(f"分析完成: {done}")
            if done:
                time.sleep(1)
                # 滚动到 AI 行情解读卡片并截图
                drv.execute_script("""
                  const cards = [...document.querySelectorAll('section .mac-window')];
                  const card = cards.find(c => c.textContent && c.textContent.includes('AI 行情解读'));
                  if (card) card.scrollIntoView({block:'center'});
                """)
                time.sleep(0.8)
                drv.save_screenshot(r"C:\Users\sbxg\Desktop\crypto_ai_trader\data\shot_ai_guards_render.png")
                # 读取卡片文本检查新区块
                txt = drv.execute_script("""
                  const cards = [...document.querySelectorAll('section .mac-window')];
                  const card = cards.find(c => c.textContent && c.textContent.includes('AI 行情解读'));
                  return card ? card.innerText.slice(0, 800) : 'NO CARD';
                """)
                print("卡片内容:")
                print(txt)
        else:
            print("未找到『立即解读』按钮")
        logs = drv.execute_script("return window.__e || []")
        print(f"console errors: {len(logs)}")
        for l in logs[:6]:
            print("  ERR:", l[:200])
    finally:
        drv.quit()


if __name__ == "__main__":
    main()

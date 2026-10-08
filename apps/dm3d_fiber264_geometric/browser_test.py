"""Browser-level verification of the offline geometric emulator (Playwright + Chromium)."""
from pathlib import Path
from playwright.sync_api import sync_playwright
import shutil

APP = Path(__file__).with_name("index.html")
with sync_playwright() as pw:
    chrome = shutil.which("chromium") or shutil.which("chromium-browser")
    options = {
        "headless": True,
        "args": ["--disable-dev-shm-usage", "--no-sandbox"],
    }
    if chrome:
        options["executable_path"] = chrome
    browser = pw.chromium.launch(**options)
    page = browser.new_page(viewport={"width": 1536, "height": 920})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.set_content(APP.read_text(encoding="utf-8"), wait_until="load")
    page.wait_for_timeout(500)
    assert page.locator("#hiddenIndex").inner_text() == "752,001"
    assert page.locator("#requiredK").inner_text() == "62"
    assert page.locator("#ratio").inner_text() == "1.61×10^5,999,998"
    assert page.locator("canvas").count() > 0
    page.locator("#orbitOne").click()
    assert page.locator("#orbitCount").inner_text() == "1"
    page.locator("#viewMode").select_option("collapse")
    assert "CONSTANT COLLAPSE" in page.locator("#modeTitle").inner_text()
    page.locator("#viewMode").select_option("latent")
    page.locator("#foldRun").click()
    page.wait_for_function(
        "document.querySelector('#converged').textContent.startsWith('YES')",
        timeout=15000,
    )
    assert page.locator("#foldCount").inner_text() == "62"
    page.locator("#workload").select_option("independent")
    assert page.locator("#ratio").inner_text() == "No equivalence established"
    page.locator("#verifySmall").click()
    assert "CTR PASS" in page.locator("#log").inner_text()
    page.locator("#verifyAll").click()
    page.wait_for_function(
        "document.querySelector('#footerStatus').textContent.includes('FULL 264³ DOMAIN VERIFIED')",
        timeout=120000,
    )
    assert page.locator("#status").inner_text() == "● FULL ENUMERATION VERIFIED"
    assert not errors, repr(errors)
    print("PASS desktop orbit, collapse, 512D contraction, workload controls, CTR and 18,399,744-state enumeration")
    mobile = browser.new_page(viewport={"width": 390, "height": 844})
    mobile_errors = []
    mobile.on("pageerror", lambda error: mobile_errors.append(str(error)))
    mobile.set_content(APP.read_text(encoding="utf-8"), wait_until="load")
    mobile.wait_for_timeout(350)
    assert mobile.locator("#requiredK").inner_text() == "62"
    mobile.locator("#orbitOne").click()
    assert mobile.locator("#orbitCount").inner_text() == "1"
    assert not mobile_errors, repr(mobile_errors)
    print("PASS 390x844 mobile layout, orbit controls, no uncaught exceptions")
    mobile.close()
    browser.close()

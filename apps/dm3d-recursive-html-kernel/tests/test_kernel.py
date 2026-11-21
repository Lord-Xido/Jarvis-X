#!/usr/bin/env python3
"""End-to-end browser regression for the standalone DM3D recursive HTML kernel."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from playwright.sync_api import sync_playwright

APP = Path(__file__).resolve().parents[1] / "index.html"
EXPECTED_SHA256 = "c103226b811467ba0a84eb0dac183d8edeeb90cca76bd0fbcc6f2490c27e8ed6"
EXPECTED_BYTES = 151091


def syntax_check(source):
    """Validate the inline scripts; the genome metadata tag is JSON, not JavaScript."""
    found = 0
    for m in re.finditer(r"<script\b([^>]*)>(.*?)</script\s*>", source, re.I | re.S):
        attrs, body = m.groups()
        if "application/json" in attrs.lower():
            json.loads(body)
            continue
        found += 1
        if shutil.which("node"):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "script.js"
                p.write_text(body, encoding="utf-8")
                subprocess.run(["node", "--check", str(p)], check=True, capture_output=True)
    assert found >= 7, "Unexpected number of executable script blocks"
    return found


def main():
    source_bytes = APP.read_bytes()
    source = source_bytes.decode("utf-8")
    assert len(source_bytes) == EXPECTED_BYTES, "Source size changed: review baseline"
    assert hashlib.sha256(source_bytes).hexdigest() == EXPECTED_SHA256, "Source SHA changed: review baseline"
    scripts = syntax_check(source)
    results = {"source_sha256": EXPECTED_SHA256, "scripts_checked": scripts}

    with sync_playwright() as playwright:
        launch = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if os.getenv("DM3D_CHROMIUM_EXECUTABLE"):
            launch["executable_path"] = os.environ["DM3D_CHROMIUM_EXECUTABLE"]
        browser = playwright.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1280, "height": 850})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.set_content(source, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_function("window.DM3DSelf && window.DM3DKernel && window.DM3DKernel.getState()==='ONLINE'", timeout=30000)
        results["boot"] = "PASS"

        actual = page.evaluate("window.DM3DSelf.sha256(new TextEncoder().encode('abc'))")
        assert actual == hashlib.sha256(b"abc").hexdigest()
        kernel = page.evaluate("async () => await window.DM3DKernel.test()")
        assert kernel["pass"] and kernel["verified"] and kernel["corruptionRejected"], kernel
        results["kernel"] = {"pass": True, "cycles": kernel["cycles"]}

        media = page.evaluate("""async () => {
          const output=[];
          for (const mode of ['text','image','audio','video','binary']) {
            const program='GENERATE '+mode.toUpperCase()+'\\nENCODE /output/test.dmxm\\nDECODE /output/test.dmxm\\nVERIFY\\nHALT';
            const p=window.DM3DKernel.newProcess(program, 'ci '+mode);
            const start=performance.now();
            while (['READY','RUNNING','WAIT'].includes(p.state) && performance.now()-start < 30000)
              await new Promise(r=>setTimeout(r,15));
            output.push({mode, state:p.state, verified:p.verified, original:p.source?.length, decoded:p.decoded?.length, error:p.error});
          }
          return output;
        }""")
        assert len(media) == 5 and all(x["state"] == "HALTED" and x["verified"] and x["original"] == x["decoded"] for x in media), media
        results["modalities"] = media

        page.locator("#kSample").click()
        page.locator("#kRun").click()
        page.wait_for_function("window.DM3DKernel.ps().some(p=>p.cycles===1000000 && p.state==='HALTED')", timeout=180000)
        audit = page.evaluate("window.DM3DKernel.audit()")
        assert audit["cpu"]["verifiedCodecCycles"] >= 1000000, audit["cpu"]
        assert sum(proc["failed"] for proc in audit["processes"]) == 0, audit["processes"]
        results["million_cycle_kernel"] = audit["cpu"]["verifiedCodecCycles"]

        probe = page.evaluate("async () => await window.DM3DSelf.testStandalone()")
        assert probe["ok"] and probe["checks"] >= 4, probe
        results["isolated_self_boot"] = "PASS"

        page.locator("#selfImport").set_input_files(str(APP))
        page.wait_for_function("window.DM3DSelf.getState().sourceMethod==='Exact imported HTML file'")
        assert page.evaluate("window.DM3DSelf.getState().sha256") == EXPECTED_SHA256
        page.locator("#selfRounds").fill("2")
        page.locator("#selfMode").select_option("verify")
        inward = page.evaluate("async () => await window.DM3DSelf.run()")
        assert inward["roundTripPasses"] == 2, inward
        assert all(x["sha256"] == EXPECTED_SHA256 for x in inward["records"]), inward["records"]
        assert page.evaluate("window.DM3DSelf.getSource()") == source
        results["self_rounds"] = 2

        corrupted_rejected = page.evaluate("""async () => {
          const a = window.DM3DSelf.getArchive().slice();
          a[a.length-1] ^= 1;
          try { await window.DMXM.decode(a); return false; } catch(e) { return true; }
        }""")
        assert corrupted_rejected, "Corrupted binary archive was accepted"

        changed_rejected = page.evaluate("""async () => {
          const original=window.DM3DSelf.getSource();
          const broken=original.replace('root.DM3DCore={AXIS,AXIS_BITS', 'root.DM3DCore={AXIS:1,AXIS_BITS');
          try { await window.DM3DSelf.sandboxProbe(broken); return false; }
          catch (e) { return true; }
        }""")
        assert changed_rejected, "Broken executable candidate was accepted"
        results["negative_tests"] = "PASS"

        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1"), "Mobile layout overflow"
        assert not errors, errors
        results["mobile"] = "PASS"
        results["page_errors"] = errors
        browser.close()

    print(json.dumps({"status": "PASS", "results": results}, indent=2))


if __name__ == "__main__":
    main()

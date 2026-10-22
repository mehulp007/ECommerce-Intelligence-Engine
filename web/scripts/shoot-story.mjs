// Dev tool: capture each story scene with the installed Chrome.
// Usage: node scripts/shoot-story.mjs <out-dir> [base-url] [width] [height]
import { mkdirSync } from "node:fs";
import { chromium } from "@playwright/test";

const [out = "story-shots", base = "http://127.0.0.1:3000", width = "1440", height = "900"] = process.argv.slice(2);
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({
  channel: "chrome",
  args: ["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"],
});
const page = await browser.newPage({ viewport: { width: Number(width), height: Number(height) } });
await page.goto(`${base}/story?capture=1`, { waitUntil: "networkidle" });
await page.waitForTimeout(4000);
const stops = [[0, 0.3], [1, 0.2], [1, 0.75], [2, 0.3], [2, 0.85], [3, 0.3], [3, 0.85], [4, 0.6], [5, 0.85], [6, 0.6]];
for (const [scene, at] of stops) {
  await page.evaluate(([i, f]) => {
    const s = document.querySelectorAll("main > section")[i];
    const top = s.getBoundingClientRect().top + scrollY;
    scrollTo(0, top + f * Math.max(0, s.offsetHeight - innerHeight));
  }, [scene, at]);
  await page.waitForTimeout(2200);
  await page.screenshot({ path: `${out}/scene-${scene + 1}-${Math.round(at * 100)}.png` });
}
const gl = await page.evaluate(() => {
  const c = document.createElement("canvas").getContext("webgl2");
  const info = c?.getExtension("WEBGL_debug_renderer_info");
  return info ? c.getParameter(info.UNMASKED_RENDERER_WEBGL) : "unknown";
});
console.log("renderer:", gl);
await browser.close();

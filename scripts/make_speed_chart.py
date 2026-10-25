"""Render docs/media/speed-projection.{html,png} from the run data."""
import subprocess, sys
OUT = "/home/user/cirrus-s-1a/docs/media/speed-projection"
versions = [("v0", "CUDA cores"), ("v1", "tensor cores"), ("v2", "pipelined"),
            ("v3", "latency-lean"), ("v4", "all 48 SMs"), ("v5", "lean loads")]
t25 = [416, 2978, 2815, 3945, 4222, 4372]   # tau 2.5, verify 8, 32K context
t165 = [516, 2190, 2026, 2583, 2819, 2955]  # tau 1.65, verify 4, 32K context
W, H = 1200, 675
X0, X1 = 150, 980                     # first / last category
YT, YB, VMAX = 196, 556, 5000         # plot top/bottom and value range
xs = [X0 + i * (X1 - X0) / 5 for i in range(6)]
y = lambda v: YB - v * (YB - YT) / VMAX
C = {"surface": "#1a1a19", "t1": "#f0efec", "t2": "#c3c2b7", "t3": "#8c8b85", "grid": "#2e2e2b",
     "axis": "#4a4a46", "s1": "#3987e5", "s2": "#d95926"}

def path(vals):
    return "M" + " L".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, vals))

def fmt(v):
    return f"{v:,}"

svg = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
# gridlines + y labels
for v in range(0, VMAX + 1, 1000):
    stroke = C["axis"] if v == 0 else C["grid"]
    svg.append(f'<line x1="96" x2="1152" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{stroke}" stroke-width="1"/>')
    svg.append(f'<text x="84" y="{y(v) + 5:.1f}" text-anchor="end" class="tick">{fmt(v)}</text>')
# reference lines
for v, label, dash in ((4000, "4,000 tok/s", "6 6"), (3000, "3,000 tok/s original target", "2 6")):
    svg.append(f'<line x1="96" x2="1152" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{C["t2"]}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
    svg.append(f'<text x="104" y="{y(v) - 9:.1f}" class="ref">{label}</text>')
# series lines then markers (2px surface ring)
for vals, col in ((t165, C["s2"]), (t25, C["s1"])):
    svg.append(f'<path d="{path(vals)}" fill="none" stroke="{col}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
    for x, v in zip(xs, vals):
        svg.append(f'<circle cx="{x:.1f}" cy="{y(v):.1f}" r="6" fill="{col}" stroke="{C["surface"]}" stroke-width="2.5"/>')
# direct end labels (text in text tokens; the marker carries identity)
for vals, tau, dy in ((t25, "τ = 2.5", 2), (t165, "τ = 1.65", 30)):  # 1.65 sits under the 3,000 line
    yy = y(vals[-1])
    svg.append(f'<text x="{X1 + 20}" y="{yy + dy:.1f}" class="endv">{fmt(vals[-1])}</text>')
    svg.append(f'<text x="{X1 + 20}" y="{yy + dy + 20:.1f}" class="endl">tok/s at {tau}</text>')
# start labels for scale of the climb
svg.append(f'<text x="{xs[0]}" y="{y(t165[0]) + 30:.1f}" text-anchor="middle" class="startl">~420-520</text>')
# x labels
for x, (v, d) in zip(xs, versions):
    svg.append(f'<text x="{x:.1f}" y="{YB + 30}" text-anchor="middle" class="xv">{v}</text>')
    svg.append(f'<text x="{x:.1f}" y="{YB + 50}" text-anchor="middle" class="xd">{d}</text>')
svg.append("</svg>")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Decode speed projection</title>
<style>
html,body{{margin:0;background:{C['surface']};}}
.card{{width:{W}px;height:{H}px;position:relative;background:{C['surface']};font-family:Inter,sans-serif;color:{C['t1']};overflow:hidden}}
.eyebrow{{position:absolute;left:48px;top:40px;font-size:15px;color:{C['t2']};letter-spacing:.02em}}
.title{{position:absolute;left:48px;top:64px;font-family:'Inter Display',Inter,sans-serif;font-size:36px;font-weight:700;letter-spacing:-.01em}}
.legend{{position:absolute;left:48px;top:122px;font-size:15px;color:{C['t2']};display:flex;gap:22px}}
.legend span{{display:inline-flex;align-items:center;gap:8px}}
.dot{{width:12px;height:12px;border-radius:50%;display:inline-block}}
.hero{{position:absolute;right:48px;top:30px;text-align:right}}
.hero .n{{font-family:'Inter Display',Inter,sans-serif;font-size:76px;font-weight:700;letter-spacing:-.02em;line-height:1}}
.hero .u{{font-size:16px;color:{C['t2']};margin-top:6px}}
.hero .s{{font-size:14px;color:{C['t3']};margin-top:4px}}
svg{{position:absolute;left:0;top:0}}
.tick{{font-size:13px;fill:{C['t3']};font-variant-numeric:tabular-nums}}
.ref{{font-size:13px;fill:{C['t2']};font-weight:500}}
.endv{{font-size:22px;font-weight:700;fill:{C['t1']};font-variant-numeric:tabular-nums}}
.endl{{font-size:13px;fill:{C['t2']}}}
.startl{{font-size:13px;fill:{C['t3']}}}
.xv{{font-size:15px;font-weight:600;fill:{C['t1']}}}
.xd{{font-size:13px;fill:{C['t2']}}}
.foot{{position:absolute;left:48px;right:48px;bottom:22px;font-size:12.5px;line-height:1.45;color:{C['t3']}}}
</style></head><body><div class="card">
{''.join(svg)}
<div class="eyebrow">155M-parameter model · one NVIDIA DGX Spark</div>
<div class="title">Projected decode speed, kernel by kernel</div>
<div class="legend"><span><i class="dot" style="background:{C['s1']}"></i>τ = 2.5 accepted tokens per step</span>
<span><i class="dot" style="background:{C['s2']}"></i>τ = 1.65 (conservative)</span></div>
<div class="hero"><div class="n">4,372</div><div class="u">tok/s projected · τ = 2.5 · 32K context</div>
<div class="s">weight stream per step: 963 µs → 490 µs</div></div>
<div class="foot">Projection, not an end-to-end run. The full 90.8 MB decode-step weight stream (4-bit NVFP4) was measured on a DGX Spark
with random weights at the real model shapes; the remaining KV/state traffic is modeled. τ (accepted tokens per speculative-decoding
step) is assumed until measured. The model is not trained yet.</div>
</div></body></html>"""
open(OUT + ".html", "w").write(html)
chrome = "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell"
subprocess.run([chrome, "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT}.png", f"file://{OUT}.html"],
               check=True, capture_output=True, timeout=120)
print("ok")

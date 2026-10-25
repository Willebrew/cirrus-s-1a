"""Render docs/media/speed-projection.{html,png} from the run data."""
import subprocess
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
# black and white: identity is carried by stroke style and marker fill, not hue
C = {"surface": "#000000", "t1": "#ffffff", "t2": "#b4b4b4", "t3": "#6e6e6e", "grid": "#1c1c1c",
     "axis": "#5a5a5a", "s1": "#ffffff", "s2": "#9a9a9a"}
SERIES = (  # values, color, dash, filled marker
    (t165, C["s2"], "8 5", False),
    (t25, C["s1"], None, True),
)

def path(vals):
    return "M" + " L".join(f"{x:.1f},{y(v):.1f}" for x, v in zip(xs, vals))

def fmt(v):
    return f"{v:,}"

def marker(x, yy, col, filled, s=10):
    fill = col if filled else C["surface"]
    return (f'<rect x="{x - s / 2:.1f}" y="{yy - s / 2:.1f}" width="{s}" height="{s}" fill="{fill}" '
            f'stroke="{col}" stroke-width="2" shape-rendering="crispEdges"/>')

svg = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
# hairline grid + y labels
for v in range(0, VMAX + 1, 1000):
    stroke = C["axis"] if v == 0 else C["grid"]
    svg.append(f'<line x1="96" x2="1152" y1="{y(v):.0f}" y2="{y(v):.0f}" stroke="{stroke}" stroke-width="1" '
               f'shape-rendering="crispEdges"/>')
    svg.append(f'<text x="84" y="{y(v) + 4:.1f}" text-anchor="end" class="tick">{fmt(v)}</text>')
# x tick marks on the baseline
for x in xs:
    svg.append(f'<line x1="{x:.0f}" x2="{x:.0f}" y1="{YB}" y2="{YB + 8}" stroke="{C["axis"]}" stroke-width="1" '
               f'shape-rendering="crispEdges"/>')
# reference lines
for v, label, dash in ((4000, "4,000 TOK/S", "4 4"), (3000, "3,000 TOK/S  ORIGINAL TARGET", "1 5")):
    svg.append(f'<line x1="96" x2="1152" y1="{y(v):.0f}" y2="{y(v):.0f}" stroke="{C["t2"]}" stroke-width="1" '
               f'stroke-dasharray="{dash}" shape-rendering="crispEdges"/>')
    svg.append(f'<text x="104" y="{y(v) - 9:.1f}" class="ref">{label}</text>')
# series lines then square markers
for vals, col, dash, filled in SERIES:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<path d="{path(vals)}" fill="none" stroke="{col}" stroke-width="2.5" stroke-linejoin="miter" '
               f'stroke-linecap="butt"{d}/>')
    for x, v in zip(xs, vals):
        svg.append(marker(x, y(v), col, filled))
# direct end labels (text in text tokens; the marker carries identity)
# 2.5: caption above the value so both clear the 4,000 line; 1.65: both below the 3,000 line
yy = y(t25[-1])
svg.append(f'<text x="{X1 + 22}" y="{yy - 20:.1f}" class="endl">TOK/S AT τ = 2.5</text>')
svg.append(f'<text x="{X1 + 22}" y="{yy + 9:.1f}" class="endv">{fmt(t25[-1])}</text>')
yy = y(t165[-1])
svg.append(f'<text x="{X1 + 22}" y="{yy + 36:.1f}" class="endv2">{fmt(t165[-1])}</text>')
svg.append(f'<text x="{X1 + 22}" y="{yy + 56:.1f}" class="endl">TOK/S AT τ = 1.65</text>')
# start label for scale of the climb
svg.append(f'<text x="{xs[0]}" y="{y(t165[0]) + 30:.1f}" text-anchor="middle" class="startl">~420-520</text>')
# x labels
for x, (v, d) in zip(xs, versions):
    svg.append(f'<text x="{x:.1f}" y="{YB + 32}" text-anchor="middle" class="xv">{v.upper()}</text>')
    svg.append(f'<text x="{x:.1f}" y="{YB + 52}" text-anchor="middle" class="xd">{d}</text>')
svg.append("</svg>")

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Decode speed projection</title>
<style>
html,body{{margin:0;background:{C['surface']};}}
.card{{width:{W}px;height:{H}px;position:relative;background:{C['surface']};font-family:Inter,sans-serif;color:{C['t1']};overflow:hidden;
  -webkit-font-smoothing:antialiased}}
.frame{{position:absolute;inset:20px;border:1px solid #262626}}
.title{{position:absolute;left:48px;top:48px;font-family:'Inter Display',Inter,sans-serif;font-size:38px;font-weight:800;letter-spacing:-.03em;line-height:1}}
.legend{{position:absolute;left:48px;top:108px;font-family:'JetBrains Mono',monospace;font-size:13px;color:{C['t2']};display:flex;gap:28px;letter-spacing:.02em}}
.legend span{{display:inline-flex;align-items:center;gap:10px}}
.sw{{display:inline-block;width:30px;height:0;position:relative}}
.sw.solid{{border-top:2.5px solid {C['s1']}}}
.sw.dash{{border-top:2.5px dashed {C['s2']}}}
.sw i{{position:absolute;left:10px;top:-6.25px;width:10px;height:10px;box-sizing:border-box;border:2px solid}}
.sw.solid i{{background:{C['s1']};border-color:{C['s1']}}}
.sw.dash i{{background:{C['surface']};border-color:{C['s2']}}}
.hero{{position:absolute;right:48px;top:36px;text-align:right}}
.hero .n{{font-family:'Inter Display',Inter,sans-serif;font-size:84px;font-weight:800;letter-spacing:-.045em;line-height:.9;font-variant-numeric:tabular-nums}}
.hero .u{{font-family:'JetBrains Mono',monospace;font-size:13px;color:{C['t2']};margin-top:12px;letter-spacing:.04em}}
.hero .s{{font-family:'JetBrains Mono',monospace;font-size:12px;color:{C['t3']};margin-top:5px;letter-spacing:.02em}}
svg{{position:absolute;left:0;top:0}}
.tick{{font-family:'JetBrains Mono',monospace;font-size:12px;fill:{C['t3']}}}
.ref{{font-family:'JetBrains Mono',monospace;font-size:11.5px;fill:{C['t2']};font-weight:500;letter-spacing:.08em}}
.endv{{font-family:'Inter Display',Inter,sans-serif;font-size:26px;font-weight:800;fill:{C['t1']};letter-spacing:-.02em}}
.endv2{{font-family:'Inter Display',Inter,sans-serif;font-size:26px;font-weight:800;fill:{C['t2']};letter-spacing:-.02em}}
.endl{{font-family:'JetBrains Mono',monospace;font-size:11.5px;fill:{C['t2']};letter-spacing:.06em}}
.startl{{font-family:'JetBrains Mono',monospace;font-size:12px;fill:{C['t3']}}}
.xv{{font-family:'JetBrains Mono',monospace;font-size:14px;font-weight:700;fill:{C['t1']};letter-spacing:.04em}}
.xd{{font-family:'JetBrains Mono',monospace;font-size:11.5px;fill:{C['t2']}}}
</style></head><body><div class="card">
<div class="frame"></div>
{''.join(svg)}
<div class="title">Projected decode speed,<br>kernel by kernel</div>
<div class="legend" style="top:140px"><span><i class="sw solid"><i></i></i>τ = 2.5 accepted tokens/step</span>
<span><i class="sw dash"><i></i></i>τ = 1.65 conservative</span></div>
<div class="hero"><div class="n">4,372</div><div class="u">TOK/S PROJECTED · τ = 2.5 · 32K CONTEXT</div>
<div class="s">weight stream per step: 963 µs → 490 µs</div></div>
</div></body></html>"""
open(OUT + ".html", "w").write(html)
chrome = "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell"
subprocess.run([chrome, "--no-sandbox", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT}.png", f"file://{OUT}.html"],
               check=True, capture_output=True, timeout=120)
print("ok")

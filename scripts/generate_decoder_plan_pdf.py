"""把 decoder 重构方案 Markdown 渲染为 PDF。

输入: docs/decoder_redesign_plan.md
输出: docs/decoder_redesign_plan.pdf

复用 generate_review_speech_pdf.py 的渲染器;支持 ``` 代码块。
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


mpl.rcParams["font.sans-serif"] = [
    "Hiragino Sans GB", "STHeiti", "Songti SC",
    "PingFang HK", "Heiti TC", "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "decoder_redesign_plan.md"
OUT = ROOT / "docs" / "decoder_redesign_plan.pdf"

PAGE_W, PAGE_H = 8.27, 11.69
TOP, BOTTOM, LEFT, RIGHT = 0.95, 0.06, 0.07, 0.93

C_TITLE = "#1F3A5F"
C_H2    = "#2E5A88"
C_H3    = "#C0392B"
C_BODY  = "#222222"
C_GRAY  = "#666666"
C_BG_QUOTE  = "#FEF6E6"
C_BG_TABLE_HDR = "#EBF3FA"
C_BG_TABLE_ALT = "#F7F9FB"
C_BG_CODE = "#F4F4F4"


def strip_md(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"(?<![*\w])\*([^*]+)\*(?![*\w])", r"\1", text)
    return text


def wrap_zh(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    lines = []
    cur = ""
    width = 0.0
    for ch in text:
        w = 1.0 if ord(ch) > 127 else 0.55
        if width + w > max_chars and cur:
            lines.append(cur)
            cur = ""
            width = 0.0
        cur += ch
        width += w
    if cur:
        lines.append(cur)
    return lines


class PdfRenderer:
    def __init__(self, pdf: PdfPages):
        self.pdf = pdf
        self._new_page()

    def _new_page(self):
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, 1); self.ax.set_ylim(0, 1)
        self.ax.axis("off")
        self.y = TOP

    def _flush(self):
        self.pdf.savefig(self.fig)
        plt.close(self.fig)

    def _ensure(self, h):
        if self.y - h < BOTTOM:
            self._flush()
            self._new_page()

    def text(self, txt, *, fontsize=10, color=C_BODY, bold=False,
             italic=False, x=LEFT, line_h=0.020, max_chars=46):
        if not txt.strip():
            self._ensure(0.012)
            self.y -= 0.012
            return
        lines = wrap_zh(txt, max_chars)
        for line in lines:
            self._ensure(line_h)
            self.ax.text(x, self.y, line,
                         fontsize=fontsize, color=color,
                         fontweight="bold" if bold else "normal",
                         style="italic" if italic else "normal",
                         va="top", linespacing=1.4)
            self.y -= line_h

    def h1(self, txt):
        self._ensure(0.06)
        self.ax.add_patch(Rectangle(
            (0, self.y - 0.04), 1, 0.04,
            transform=self.ax.transAxes, facecolor=C_TITLE,
            edgecolor="none",
        ))
        self.ax.text(0.5, self.y - 0.020, strip_md(txt),
                     fontsize=15, color="white", fontweight="bold",
                     ha="center", va="center")
        self.y -= 0.055

    def h2(self, txt):
        self._ensure(0.045)
        self.y -= 0.015
        self._ensure(0.030)
        self.ax.text(LEFT, self.y, strip_md(txt),
                     fontsize=13, color=C_H2, fontweight="bold", va="top")
        self.y -= 0.028
        self.ax.plot([LEFT, RIGHT], [self.y + 0.005, self.y + 0.005],
                     transform=self.ax.transAxes, color=C_H2, lw=0.8)
        self.y -= 0.005

    def h3(self, txt):
        self._ensure(0.030)
        self.y -= 0.005
        self.ax.text(LEFT, self.y, strip_md(txt),
                     fontsize=11.5, color=C_H3, fontweight="bold", va="top")
        self.y -= 0.025

    def paragraph(self, txt):
        if not txt.strip():
            self.y -= 0.008
            return
        self.text(strip_md(txt), fontsize=10, color=C_BODY,
                  x=LEFT, line_h=0.018, max_chars=46)
        self.y -= 0.005

    def quote(self, txt):
        clean = strip_md(txt)
        lines = wrap_zh(clean, 44)
        h = 0.022 * len(lines) + 0.012
        self._ensure(h)
        self.ax.add_patch(Rectangle(
            (LEFT - 0.005, self.y - h), RIGHT - LEFT + 0.01, h,
            transform=self.ax.transAxes, facecolor=C_BG_QUOTE,
            edgecolor="none", alpha=0.7,
        ))
        self.ax.add_patch(Rectangle(
            (LEFT - 0.005, self.y - h), 0.005, h,
            transform=self.ax.transAxes, facecolor=C_H3,
            edgecolor="none",
        ))
        yy = self.y - 0.010
        for line in lines:
            self.ax.text(LEFT + 0.012, yy, line, fontsize=9.5,
                         color=C_GRAY, va="top", style="italic")
            yy -= 0.020
        self.y -= h
        self.y -= 0.008

    def code_block(self, lines: list[str]):
        line_h = 0.0175
        h = line_h * len(lines) + 0.012
        self._ensure(h + 0.005)
        self.ax.add_patch(Rectangle(
            (LEFT - 0.005, self.y - h), RIGHT - LEFT + 0.01, h,
            transform=self.ax.transAxes, facecolor=C_BG_CODE,
            edgecolor="#DDD", lw=0.4,
        ))
        yy = self.y - 0.010
        for ln in lines:
            self.ax.text(LEFT + 0.005, yy, ln,
                         fontsize=8.5, color="#1A1A1A",
                         va="top", family="monospace")
            yy -= line_h
        self.y -= h
        self.y -= 0.008

    def bullet(self, txt, *, level=0):
        indent = LEFT + 0.025 * (level + 1)
        max_chars = 44 - int(level * 3)
        clean = strip_md(txt)
        lines = wrap_zh(clean, max_chars)
        for i, line in enumerate(lines):
            self._ensure(0.020)
            if i == 0:
                self.ax.text(indent - 0.018, self.y, "•",
                             fontsize=11, color=C_H3,
                             fontweight="bold", va="top")
            self.ax.text(indent, self.y, line,
                         fontsize=10, color=C_BODY, va="top",
                         linespacing=1.4)
            self.y -= 0.020

    def numbered(self, num, txt):
        clean = strip_md(txt)
        indent = LEFT + 0.030
        lines = wrap_zh(clean, 44)
        for i, line in enumerate(lines):
            self._ensure(0.020)
            if i == 0:
                self.ax.text(LEFT + 0.005, self.y, f"{num}.",
                             fontsize=10, color=C_H3,
                             fontweight="bold", va="top")
            self.ax.text(indent, self.y, line,
                         fontsize=10, color=C_BODY, va="top")
            self.y -= 0.020

    def hr(self):
        self._ensure(0.025)
        self.y -= 0.010
        self.ax.plot([LEFT, RIGHT], [self.y, self.y],
                     transform=self.ax.transAxes,
                     color=C_GRAY, lw=0.4, ls="--", alpha=0.6)
        self.y -= 0.013

    def table(self, rows):
        if not rows:
            return
        ncol = len(rows[0])
        widths = []
        for c in range(ncol):
            mx = max(len(strip_md(r[c])) if c < len(r) else 0 for r in rows)
            widths.append(max(mx, 4))
        total = sum(widths)
        col_widths = [w / total for w in widths]
        col_x = [LEFT]
        for w in col_widths[:-1]:
            col_x.append(col_x[-1] + w * (RIGHT - LEFT))

        line_h = 0.020
        row_heights = []
        for r in rows:
            max_lines = 1
            for c, txt in enumerate(r):
                if c >= ncol:
                    break
                col_chars = max(int(col_widths[c] * 51), 6)
                lines = wrap_zh(strip_md(txt), max(col_chars - 2, 8))
                max_lines = max(max_lines, len(lines))
            row_heights.append(max(line_h, max_lines * line_h * 0.95))

        total_h = sum(row_heights) + 0.005
        # 表格不分页:如果当前页放不下则换页
        if self.y - (total_h + 0.01) < BOTTOM:
            self._flush()
            self._new_page()
        self.y -= 0.005

        for ri, (r, rh) in enumerate(zip(rows, row_heights)):
            if ri == 0:
                bg = C_BG_TABLE_HDR
            else:
                bg = C_BG_TABLE_ALT if ri % 2 == 0 else "white"
            self.ax.add_patch(Rectangle(
                (LEFT, self.y - rh), RIGHT - LEFT, rh,
                transform=self.ax.transAxes, facecolor=bg,
                edgecolor="#DDD", lw=0.4,
            ))
            for c in range(min(ncol, len(r))):
                col_chars = max(int(col_widths[c] * 51), 6)
                lines = wrap_zh(strip_md(r[c]), max(col_chars - 2, 8))
                yy = self.y - 0.010
                for ln in lines:
                    self.ax.text(col_x[c] + 0.005, yy, ln,
                                 fontsize=9 if ri > 0 else 9.5,
                                 color=C_H2 if ri == 0 else C_BODY,
                                 fontweight="bold" if ri == 0 else "normal",
                                 va="top")
                    yy -= line_h * 0.95
            self.y -= rh
        self.y -= 0.008

    def finish(self):
        self._flush()


def parse_markdown(text: str):
    """简单解析 markdown 为 (kind, content) 列表。支持 ``` 代码块。"""
    lines = text.splitlines()
    items = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 代码块
        if stripped.startswith("```"):
            j = i + 1
            code_lines = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code_lines.append(lines[j])
                j += 1
            items.append(("code", code_lines))
            i = j + 1
            continue

        if not stripped:
            items.append(("blank", ""))
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            level = len(m.group(1))
            items.append((f"h{level}", m.group(2)))
            i += 1
            continue

        if stripped.startswith(">"):
            quote = stripped[1:].strip()
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith(">"):
                quote += " " + lines[j].strip()[1:].strip()
                j += 1
            items.append(("quote", quote))
            i = j
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in row if c):
                    i += 1
                    continue
                tbl.append(row)
                i += 1
            items.append(("table", tbl))
            continue

        if stripped in {"---", "***", "___"}:
            items.append(("hr", ""))
            i += 1
            continue

        m = re.match(r"^(\d+)\.\s+(.*)", stripped)
        if m:
            items.append(("numbered", (m.group(1), m.group(2))))
            i += 1
            continue

        m = re.match(r"^([-*])\s+(.*)", stripped)
        if m:
            items.append(("bullet", m.group(2)))
            i += 1
            continue

        items.append(("p", stripped))
        i += 1

    return items


def main():
    src = SRC.read_text(encoding="utf-8")
    items = parse_markdown(src)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT) as pdf:
        r = PdfRenderer(pdf)
        for kind, content in items:
            if kind == "h1":
                r.h1(content)
            elif kind == "h2":
                r.h2(content)
            elif kind == "h3":
                r.h3(content)
            elif kind in ("h4", "h5", "h6"):
                r.h3(content)
            elif kind == "p":
                r.paragraph(content)
            elif kind == "bullet":
                r.bullet(content)
            elif kind == "numbered":
                num, txt = content
                r.numbered(num, txt)
            elif kind == "quote":
                r.quote(content)
            elif kind == "hr":
                r.hr()
            elif kind == "table":
                r.table(content)
            elif kind == "code":
                r.code_block(content)
            elif kind == "blank":
                r.y -= 0.005

        r.finish()

        info = pdf.infodict()
        info["Title"] = "Decoder 重构方案 · TokenLUT-BG"
        info["Author"] = "CS60 Group"
        info["Subject"] = "FiveK 图像增强 decoder 完整重构方案"

    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()

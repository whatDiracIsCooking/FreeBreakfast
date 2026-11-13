"""Render a rebuild design doc to a standalone, theme-aware HTML page.

usage: md2html.py <in.md> <out.html> [sibling.md ...] [figures/<doc>.figs]

Links to a listed sibling .md are rewritten to its .html. A `.figs` argument
splices in the hand-drawn SVG figures it holds (see `insert_figures`). The
.md is the source; regenerate the .html after editing either. From this
directory:

    for d in programming_model memory; do
      uv run --no-project --with markdown-it-py python md2html.py \
        $d.md $d.html programming_model.md memory.md figures/$d.figs
    done
"""

import html
import re
import sys
from pathlib import Path

from markdown_it import MarkdownIt

CSS = """
:root{--bg:#fbfaf7;--fg:#1f2328;--muted:#5d6670;--rule:#e3e0d8;--accent:#8a4b16;
--code-bg:#f1eee6;--th-bg:#f4f1ea;--link:#0b5cad;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#16181c;
--fg:#e4e6ea;--muted:#9aa3ad;--rule:#2c3038;--accent:#e0a36a;--code-bg:#22262d;
--th-bg:#1d2026;--link:#7fb4ff;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#16181c;--fg:#e4e6ea;--muted:#9aa3ad;--rule:#2c3038;
--accent:#e0a36a;--code-bg:#22262d;--th-bg:#1d2026;--link:#7fb4ff;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.65 Charter,"Bitstream Charter","Iowan Old Style",Georgia,serif;
padding-inline:16px}
.wrap{max-width:72rem;margin:0 auto;display:grid;grid-template-columns:15rem minmax(0,
1fr);
gap:3rem;padding-block:2.5rem 4rem}
nav{position:sticky;top:1.5rem;align-self:start;max-height:calc(100vh - 3rem);
overflow:auto;
font:13.5px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
nav .kicker{text-transform:uppercase;letter-spacing:.08em;font-size:11px;
color:var(--muted);
margin-bottom:.6rem}
nav ol{list-style:none;margin:0;padding:0}
nav li{margin:.3rem 0}
nav li.l3{padding-left:.9rem;font-size:12.5px}
nav a{color:var(--muted);text-decoration:none}
nav a:hover{color:var(--accent)}
nav .sib{margin-top:1.4rem;padding-top:.8rem;border-top:1px solid var(--rule)}
main{max-width:44rem}
h1,h2,h3{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.25;
letter-spacing:-.01em}
h1{font-size:2rem;margin:0 0 1.5rem}
h1 .pre{display:block;font-size:.8rem;font-weight:600;text-transform:uppercase;
letter-spacing:.1em;color:var(--accent);margin-bottom:.4rem}
h2{font-size:1.35rem;margin:3rem 0 .9rem;padding-top:1.2rem;
border-top:1px solid var(--rule)}
h3{font-size:1.08rem;margin:2.2rem 0 .7rem}
h2 a.hl,h3 a.hl{color:inherit;text-decoration:none}
p,li{hyphens:auto}
strong{font-weight:650}
a{color:var(--link);text-underline-offset:2px}
code{font:.86em/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
background:var(--code-bg);padding:.1em .3em;border-radius:3px}
pre{background:var(--code-bg);padding:1rem 1.1rem;border-radius:6px;overflow-x:auto;
line-height:1.45}
pre code{background:none;padding:0;font-size:13.5px}
.table{overflow-x:auto;margin:1.2rem 0}
table{border-collapse:collapse;font:14px/1.45 system-ui,-apple-system,"Segoe UI",
sans-serif;
min-width:100%}
th,td{border-bottom:1px solid var(--rule);padding:.5rem .7rem;text-align:left;
vertical-align:top}
th{background:var(--th-bg);font-weight:600}
td code,th code{font-size:12.5px}
ol,ul{padding-left:1.4rem}
@media (max-width:860px){.wrap{grid-template-columns:1fr;gap:0;
padding-block:1.5rem 3rem}
nav{position:static;max-height:none;margin-bottom:1rem;padding-bottom:1rem;
border-bottom:1px solid var(--rule)}}
:root{--acc-fill:rgba(138,75,22,.15);--row:#e9e5da}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--acc-fill:rgba(224,163,106,.2);--row:#2a2f37}}
:root[data-theme="dark"]{--acc-fill:rgba(224,163,106,.2);--row:#2a2f37}
figure{margin:1.8rem 0}
figure .scroll{overflow-x:auto}
figure svg{display:block;width:100%;min-width:520px;height:auto;color:var(--fg)}
figcaption{font:13px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
color:var(--muted);
margin-top:.6rem}
svg text{font:12px system-ui,-apple-system,"Segoe UI",sans-serif;fill:currentColor}
svg .t{font-weight:600}
svg .sm{font-size:11px}
svg .mt{fill:var(--muted)}
svg .at{fill:var(--accent)}
svg .ln{stroke:currentColor;fill:none;stroke-width:1.3}
svg .ml{stroke:var(--muted);fill:none;stroke-width:1}
svg .al{stroke:var(--accent);fill:none;stroke-width:1.6}
svg .dash{stroke-dasharray:5 4}
svg .box{fill:var(--code-bg);stroke:currentColor;stroke-width:1.2}
svg .row{fill:var(--row);stroke:currentColor;stroke-width:1}
svg .ab{fill:var(--acc-fill);stroke:var(--accent);stroke-width:1.5}
svg .ah{fill:var(--accent)}
svg .fh{fill:currentColor}
"""


def insert_figures(body, figs_path):
    """Splice hand-drawn figures into the rendered body.

    The file is blocks headed `@@ after|before|replace-pre <anchor>`; the
    anchor is a substring of the rendered HTML.
    """
    blocks = re.split(r"^@@ ", Path(figs_path).read_text(), flags=re.M)[1:]
    for block in blocks:
        head, fig = block.split("\n", 1)
        mode, anchor = head.split(" ", 1)
        idx = body.find(anchor)
        if idx < 0:
            sys.exit(f"figure anchor not found: {anchor!r}")
        if mode == "after":
            at = body.find("</p>", idx) + 4
            body = body[:at] + "\n" + fig + body[at:]
        elif mode == "before":
            body = body[:idx] + fig + body[idx:]
        elif mode == "replace-pre":
            start = body.rfind("<pre>", 0, idx)
            end = body.find("</pre>", idx) + 6
            body = body[:start] + fig + body[end:]
        else:
            sys.exit(f"unknown figure mode: {mode}")
    return body


def slug(text):
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s]+", "-", s)


def main():
    src, out, *rest = sys.argv[1:]
    figs = [a for a in rest if a.endswith(".figs")]
    sibs = [a for a in rest if not a.endswith(".figs")]
    md_text = Path(src).read_text()
    sib_names = {Path(s).name for s in sibs}

    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(md_text)

    toc, title = [], ""
    for i, t in enumerate(tokens):
        if t.type == "heading_open":
            inline = tokens[i + 1]
            text = inline.content
            plain = re.sub(r"`", "", text)
            if t.tag == "h1":
                title = plain
                continue
            sid = slug(plain)
            t.attrSet("id", sid)
            if t.tag in ("h2", "h3"):
                toc.append((t.tag, sid, inline))
        if t.type == "inline":
            for c in t.children or []:
                if c.type == "link_open":
                    href = c.attrGet("href")
                    if href in sib_names:
                        c.attrSet("href", href[:-3] + ".html")
        if t.type == "table_open":
            t.attrSet("data-wrap", "1")

    body = md.renderer.render(tokens, md.options, {})
    body = body.replace('<table data-wrap="1">', '<div class="table"><table>')
    body = body.replace("</table>", "</table></div>")
    for f in figs:
        body = insert_figures(body, f)
    # The h1: split "Rust rebuild: x" into a kicker and the name.
    m = re.match(r"(.*?):\s*(.*)", title)
    pre, name = (m.group(1), m.group(2)) if m else ("", title)
    body = re.sub(
        r"<h1>.*?</h1>",
        f'<h1><span class="pre">{html.escape(pre)}</span>'
        f"{html.escape(name.capitalize())}</h1>",
        body,
        count=1,
    )

    items = []
    for tag, sid, inline in toc:
        label = md.renderer.renderInline(inline.children, md.options, {})
        items.append(f'<li class="l{tag[1]}"><a href="#{sid}">{label}</a></li>')
    sib_links = "".join(
        f'<li><a href="{s[:-3]}.html">{s[:-3].replace("_", " ")}</a></li>'
        for s in sorted(sib_names)
        if s != Path(src).name
    )
    nav = (
        f'<nav><div class="kicker">Contents</div><ol>{"".join(items)}</ol>'
        + (
            '<div class="sib"><div class="kicker">See also</div>'
            f"<ol>{sib_links}</ol></div>"
            if sib_links
            else ""
        )
        + "</nav>"
    )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(name.capitalize())} — {html.escape(pre)}</title>
<!-- Generated from {Path(src).name} by md2html.py; edit the .md, not this. -->
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{nav}
<main>
{body}</main>
</div>
</body>
</html>
"""
    Path(out).write_text(page)


if __name__ == "__main__":
    main()

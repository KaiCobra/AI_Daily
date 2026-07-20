# -*- coding: utf-8 -*-
"""重建 INDEX.md:掃 papers/ 下所有解析 md,抽標題/arXiv/路徑,按月份分組。

用法:python3 tools_build_index.py
"""
import glob
import os
import re
from collections import defaultdict

GENERIC = re.compile(r"^(AI Daily|AI_Daily.*|\d{4}-\d{2}-\d{2}.*)$", re.I)


def extract_title(txt, path):
    lines = txt.splitlines()
    heads = [l for l in lines if l.startswith("#")]
    cands = []
    for h in heads[:6]:
        t = h.lstrip("# ").strip()
        t = re.sub(r"^AI Daily[::]?\s*", "", t)
        t = re.sub(r"^\d{4}-\d{2}-\d{2}\s*[::—-]?\s*", "", t)
        t = t.strip(" -—::")
        if t and not GENERIC.match(t):
            cands.append(t)
    if cands:
        return cands[0]
    m = re.search(r"\*\*論文標題\*\*[::]\s*(.+)", txt)
    if m:
        return m.group(1).strip()
    return os.path.splitext(os.path.basename(path))[0].replace("AI_Daily_", "")


def extract_arxiv(txt):
    # 優先:論文連結/arXiv 編號行;fallback:全文第一個 arxiv link
    for pat in (r"(?:論文連結|arXiv\s*(?:ID|編號)?)\*{0,2}[::]\s*.*?(\d{4}\.\d{4,5})",
                r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})"):
        m = re.search(pat, txt)
        if m:
            return m.group(1)
    return ""


def month_of(path):
    m = re.search(r"papers/\d{4}/(\d{4}-\d{2})/", path)
    if m:
        return m.group(1)
    m = re.search(r"(\d{4}-\d{2})-\d{2}", path)
    return m.group(1) if m else "misc"


def main():
    entries = []
    for p in sorted(glob.glob("papers/**/*.md", recursive=True)):
        if os.path.basename(p) == "README.md":
            continue
        txt = open(p, encoding="utf-8", errors="ignore").read()
        entries.append((month_of(p), extract_title(txt, p), extract_arxiv(txt), p))

    groups = defaultdict(list)
    for e in entries:
        groups[e[0]].append(e)

    out = ["# 論文索引(全庫)", "",
           f"共 {len(entries)} 篇解析,按月份倒序;同一論文多次研讀會出現多列。",
           "重建方式:`python3 tools_build_index.py`。", ""]
    for month in sorted(groups, reverse=True):
        out += [f"## {month}", "", "| 論文 | arXiv | 解析 |", "|---|---|---|"]
        for _, title, arxiv, p in sorted(groups[month], key=lambda x: x[3]):
            t = title.replace("|", "\\|")
            t = t if len(t) <= 90 else t[:87] + "..."
            a = f"[{arxiv}](https://arxiv.org/abs/{arxiv})" if arxiv else "—"
            name = os.path.basename(os.path.dirname(p))
            if re.match(r"^\d{4}(-\d{2})?$", name) or name == "papers":
                name = os.path.splitext(os.path.basename(p))[0]
            out.append(f"| {t} | {a} | [{name}]({p}) |")
        out.append("")
    open("INDEX.md", "w", encoding="utf-8").write("\n".join(out))
    print(f"INDEX.md: {len(entries)} entries / {len(groups)} months")


if __name__ == "__main__":
    main()

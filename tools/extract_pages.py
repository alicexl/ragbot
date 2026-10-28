"""从年报全本抽取指定页 → 精选 PDF（ragbot 语料与评测用）

用法：python -X utf8 tools/extract_pages.py <源PDF> <输出PDF> <页码列表>
页码 1-based、逗号分隔。选页原则=镜像 sample_reports/盐湖股份2024年报（精选页）.pdf
的 10 页结构：重要提示 / 释义 / 公司简介（法定代表人）/ 主要会计数据 / 非经常性
损益 / 经营讨论·行业情况 / 图表页。页面号由人工逐份年报核对后指定。
"""

import sys

import fitz


def main():
    src, out, pages = sys.argv[1], sys.argv[2], sys.argv[3]
    doc = fitz.open(src)
    sel = fitz.open()
    for p in pages.split(","):
        sel.insert_pdf(doc, from_page=int(p) - 1, to_page=int(p) - 1)
    sel.save(out)
    print(f"[OK] {out}：{len(sel)} 页（源页 {pages}）")


if __name__ == "__main__":
    main()

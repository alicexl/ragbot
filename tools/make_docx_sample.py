"""生成 docx 验收样本（sample_reports/ 下两份）

1. 星澜科技2025年度简报.docx —— 虚构公司 + 写死的数字：docx 解析管线的 smoke
   样本（段落/表格/图片三类 chunk 齐全），evalset M05 的锚定语料。标准样式排版
   （add_heading/普通表格），数字与真语料零重合。
2. 万华化学2025年度简报.docx —— 真实公司 + 真实数字（摘自万华 2025 年报 p7
   主要会计数据表），**反样式排版**：标题不用 Heading 样式（加粗大字号伪装——
   真实世界 docx 的常态）、表格带合并单元格表头。evalset M07 锚定语料，专验
   章节提取的兜底路径（样式优先、加粗+序号正则兜底）。

重新运行 = 覆盖重建（内容确定，幂等）。用法：python -X utf8 tools/make_docx_sample.py
"""

import io
import struct
import zlib
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

SAMPLE_DIR = Path(__file__).parent.parent / "sample_reports"
OUT = SAMPLE_DIR / "星澜科技2025年度简报.docx"
OUT_WANHUA = SAMPLE_DIR / "万华化学2025年度简报.docx"


def _fake_heading(doc, text: str, size: int = 12, center: bool = False):
    """反样式标题：真实文档常见的「加粗大字号伪装标题」——不用 Heading 样式。"""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p


def _png(width=240, height=120, rgb=(64, 116, 210)) -> bytes:
    """stdlib 手搓纯色 PNG——样本图只为走通「docx 图片 → VLM 描述 → image chunk」
    管线，不值得为它引 Pillow。"""
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def main():
    doc = Document()
    doc.add_heading("星澜科技2025年度经营简报", level=1)
    doc.add_paragraph(
        "星澜科技（虚构演示公司）主营工业智能传感器与边缘计算设备的研发、制造与销售，"
        "产品应用于离散制造产线的状态监测与能耗管理。本简报为 ragbot docx 导入管线的"
        "验收样本，其中数字全部虚构，与任何真实公司无关。")
    doc.add_paragraph(
        "2025年，制造业智能化改造需求持续释放，公司在汽车零部件与3C电子两个行业线"
        "均实现了客户数量与单客户采购额的双增长；海外市场方面，东南亚区域完成本地化"
        "服务网点布局，出口收入占比提升至18.4%。")

    doc.add_heading("主要财务指标", level=2)
    doc.add_paragraph("下表列示公司近三年主要财务数据（单位：元，除标注外，虚构）。")
    # 12 行指标——刻意超过 store.SMALL_TABLE_ROWS(10)，让大表格走「独立 table chunk」
    # 路径（内联小表会跟正文混切成 text chunk，evalset 的表格锚定就失配了）
    rows = [
        ("指标", "2023年度", "2024年度", "2025年度"),
        ("营业收入（元）", "612,345,678.90", "745,678,901.23", "876,543,200.55"),
        ("归属于母公司股东的净利润（元）", "58,901,234.56", "81,234,567.89", "102,345,678.90"),
        ("扣除非经常性损益的净利润（元）", "55,123,456.78", "78,456,789.01", "98,765,432.10"),
        ("经营活动产生的现金流量净额（元）", "71,234,567.89", "120,345,678.90", "158,900,000.00"),
        ("基本每股收益（元/股）", "0.61", "0.84", "1.06"),
        ("加权平均净资产收益率", "9.8%", "12.4%", "14.1%"),
        ("研发投入（元）", "61,234,567.89", "82,345,678.90", "103,432,098.76"),
        ("研发投入占营业收入比例", "10.0%", "11.0%", "11.8%"),
        ("总资产（元）", "890,123,456.78", "1,012,345,678.90", "1,145,678,901.23"),
        ("归属于上市公司股东的净资产（元）", "654,321,098.76", "723,456,789.01", "801,234,567.89"),
        ("资产负债率", "26.5%", "28.5%", "30.0%"),
        ("存货周转率（次）", "2.8", "3.1", "3.3"),
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.style = "Table Grid"
    for r, cells in enumerate(rows):
        for c, val in enumerate(cells):
            table.cell(r, c).text = val

    doc.add_paragraph("下图展示了公司近三年营业收入的增长趋势（示意图，虚构数据）。")
    doc.add_picture(io.BytesIO(_png()))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print(f"[OK] {OUT}")


def make_wanhua():
    """真实数字 + 反样式排版的 docx（evalset M07 锚定语料）。

    数字全部摘自万华化学 2025 年年度报告 p7「七、近三年主要会计数据和财务指标」，
    未抽到的格用「—」（不编造）。反样式三件套：伪标题（加粗大字号，无 Heading
    样式）、合并单元格表头、11 行表（>SMALL_TABLE_ROWS 走独立 table chunk）。
    """
    doc = Document()
    _fake_heading(doc, "万华化学2025年度经营简报", size=16, center=True)
    doc.add_paragraph(
        "本简报数据摘自万华化学集团股份有限公司2025年年度报告「七、近三年主要会计"
        "数据和财务指标」一节（单位：元，币种：人民币）。")

    _fake_heading(doc, "一、公司概况")
    doc.add_paragraph(
        "万华化学集团股份有限公司（600309.SH）总部位于山东省烟台市，是全球领先的"
        "聚氨酯（MDI、TDI）制造商，业务涵盖聚氨酯、石化、精细化学品及新兴材料"
        "四大板块。2025年公司营业收入重回两千亿元台阶，主要受益于聚氨酯板块量价"
        "齐升与石化装置负荷提升。")

    _fake_heading(doc, "二、近三年主要财务数据")
    doc.add_paragraph("下表列示公司近三年主要会计数据（期末数指标为年末值；「—」为未摘录项）。")
    rows = [
        ("近三年主要会计数据（摘自2025年年报，单位：元）", "", "", ""),   # 合并单元格标题行
        ("指标", "2023年度", "2024年度", "2025年度"),
        ("营业收入", "175,360,935,668.36", "182,069,119,160.94", "203,234,573,814.42"),
        ("利润总额", "20,009,871,906.36", "16,924,136,091.56", "16,268,157,181.59"),
        ("归属于上市公司股东的净利润", "16,815,755,534.31", "13,033,066,612.84",
         "12,527,201,094.15"),
        ("归属于上市公司股东的扣除非经常性损益的净利润", "16,438,841,961.03",
         "13,358,861,995.56", "12,143,675,765.00"),
        ("经营活动产生的现金流量净额", "26,796,752,552.51", "30,053,435,178.33",
         "33,105,189,455.82"),
        ("基本每股收益（元/股）", "—", "4.15", "3.99"),
        ("稀释每股收益（元/股）", "—", "4.15", "3.99"),
        ("归属于上市公司股东的净资产", "88,656,204,854.29", "94,625,896,066.69",
         "108,305,459,478.31"),
        ("总资产", "—", "293,333,349,744.25", "323,009,018,441.98"),
    ]
    table = doc.add_table(rows=len(rows), cols=4)
    table.style = "Table Grid"
    for r, cells in enumerate(rows):
        for c, val in enumerate(cells):
            table.cell(r, c).text = val
    table.cell(0, 0).merge(table.cell(0, 3))                 # 合并单元格表头（真实文档常态）

    _fake_heading(doc, "三、简要解读")
    doc.add_paragraph(
        "2025年公司营业收入同比增长11.62%，但归母净利润同比下降3.88%——增收不增利，"
        "主要受石化板块产品价格中枢下移与部分装置检修影响；经营活动现金流量净额"
        "同比增长10.15%，盈利质量好于账面利润表现。")

    OUT_WANHUA.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT_WANHUA))
    print(f"[OK] {OUT_WANHUA}")


if __name__ == "__main__":
    main()
    make_wanhua()

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper"
ASSET_DIR = OUT_DIR / "assets"
PDF_PATH = OUT_DIR / "TMG-Q_Research_Paper_2026.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1F5A85")
TEAL = colors.HexColor("#168C8C")
GOLD = colors.HexColor("#C28B20")
PALE_BLUE = colors.HexColor("#EAF2F8")
PALE_TEAL = colors.HexColor("#E7F5F3")
PALE_GOLD = colors.HexColor("#FFF6DF")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#5B6773")
GRID = colors.HexColor("#C9D3DC")
WHITE = colors.white

PAGE_W, PAGE_H = letter
MARGIN_X = 0.62 * inch
TOP = 0.58 * inch
BOTTOM = 0.56 * inch
GUTTER = 0.22 * inch
COL_W = (PAGE_W - 2 * MARGIN_X - GUTTER) / 2


def register_fonts():
    candidates = [
        ("Aptos", r"C:\Windows\Fonts\aptos.ttf"),
        ("Aptos-Bold", r"C:\Windows\Fonts\aptosbd.ttf"),
        ("Cambria", r"C:\Windows\Fonts\cambria.ttc"),
        ("Cambria-Bold", r"C:\Windows\Fonts\cambriab.ttc"),
    ]
    for name, path in candidates:
        if Path(path).exists():
            try:
                pdfmetrics.registerFont(TTFont(name, path))
            except Exception:
                pass


register_fonts()
BODY_FONT = "Aptos" if "Aptos" in pdfmetrics.getRegisteredFontNames() else "Helvetica"
BOLD_FONT = "Aptos-Bold" if "Aptos-Bold" in pdfmetrics.getRegisteredFontNames() else "Helvetica-Bold"
SERIF_FONT = "Cambria" if "Cambria" in pdfmetrics.getRegisteredFontNames() else "Times-Roman"
SERIF_BOLD = "Cambria-Bold" if "Cambria-Bold" in pdfmetrics.getRegisteredFontNames() else "Times-Bold"


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="PaperBody",
        fontName=SERIF_FONT,
        fontSize=8.55,
        leading=10.65,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=4.2,
        splitLongWords=True,
    )
)
styles.add(
    ParagraphStyle(
        name="PaperSmall",
        fontName=SERIF_FONT,
        fontSize=7.25,
        leading=8.8,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=2.5,
    )
)
styles.add(
    ParagraphStyle(
        name="Section",
        fontName=BOLD_FONT,
        fontSize=11.1,
        leading=13,
        textColor=NAVY,
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Subsection",
        fontName=BOLD_FONT,
        fontSize=9.2,
        leading=11,
        textColor=BLUE,
        spaceBefore=4,
        spaceAfter=2.5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        name="Caption",
        fontName=BODY_FONT,
        fontSize=7,
        leading=8.4,
        textColor=MUTED,
        alignment=TA_LEFT,
        spaceBefore=2,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        name="Abstract",
        fontName=SERIF_FONT,
        fontSize=8.8,
        leading=11.1,
        textColor=INK,
        alignment=TA_JUSTIFY,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="Keyword",
        fontName=BODY_FONT,
        fontSize=7.8,
        leading=9.5,
        textColor=MUTED,
        alignment=TA_LEFT,
        spaceAfter=4,
    )
)
styles.add(
    ParagraphStyle(
        name="Reference",
        fontName=SERIF_FONT,
        fontSize=6.7,
        leading=8.15,
        leftIndent=10,
        firstLineIndent=-10,
        textColor=INK,
        spaceAfter=2.2,
    )
)
styles.add(
    ParagraphStyle(
        name="Callout",
        fontName=BODY_FONT,
        fontSize=8,
        leading=10,
        textColor=NAVY,
        alignment=TA_LEFT,
        leftIndent=5,
        rightIndent=5,
        spaceBefore=3,
        spaceAfter=3,
    )
)


def font(size=22, bold=False):
    paths = [
        r"C:\Windows\Fonts\aptosbd.ttf" if bold else r"C:\Windows\Fonts\aptos.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def rounded_box(draw, xy, fill, outline, radius=16, width=2):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def centered(draw, xy, text, fnt, fill):
    box = draw.textbbox((0, 0), text, font=fnt)
    x = xy[0] - (box[2] - box[0]) / 2
    y = xy[1] - (box[3] - box[1]) / 2
    draw.text((x, y), text, font=fnt, fill=fill)


def make_architecture():
    img = Image.new("RGB", (1400, 520), "white")
    d = ImageDraw.Draw(img)
    title = font(34, True)
    label = font(25, True)
    small = font(21, False)
    d.text((55, 28), "TMG-Q end-to-end compression pipeline", font=title, fill="#102A43")
    boxes = [
        (45, 120, 245, 350, "#EAF2F8", "#1F5A85", "Model", ["Linear / Conv1D", "Embedding", "Tied LM head"]),
        (295, 120, 495, 350, "#FFF6DF", "#C28B20", "Calibrate", ["WikiText-2", "Activation energy", "Hessian proxy"]),
        (545, 120, 745, 350, "#E7F5F3", "#168C8C", "Probe", ["2-bit codebook", "3-bit codebook", "4-bit linear"]),
        (795, 120, 995, 350, "#EAF2F8", "#1F5A85", "Recover", ["Sparse outliers", "Optional SVD", "Vocabulary distill"]),
        (1045, 120, 1245, 350, "#E7F5F3", "#168C8C", "Pack", ["INT32 payload", "Portable .pt", "Byte accounting"]),
    ]
    for x1, y1, x2, y2, fill, outline, head, lines in boxes:
        rounded_box(d, (x1, y1, x2, y2), fill, outline, radius=18, width=3)
        centered(d, ((x1 + x2) / 2, y1 + 47), head, label, outline)
        for i, line in enumerate(lines):
            centered(d, ((x1 + x2) / 2, y1 + 105 + i * 42), line, small, "#263746")
    for x in [245, 495, 745, 995]:
        d.line((x + 8, 235, x + 42, 235), fill="#60798C", width=5)
        d.polygon([(x + 42, 235), (x + 25, 224), (x + 25, 246)], fill="#60798C")
    rounded_box(d, (330, 395, 1060, 478), "#F4F7FA", "#C9D3DC", radius=14, width=2)
    centered(
        d,
        (695, 435),
        "Validation: perplexity + full-file ratio + matched-matrix ratio + quality gate",
        font(22, True),
        "#102A43",
    )
    path = ASSET_DIR / "architecture.png"
    img.save(path, quality=95)
    return path


def make_gpt2_results():
    img = Image.new("RGB", (1200, 590), "white")
    d = ImageDraw.Draw(img)
    title = font(31, True)
    axis = font(19, False)
    value = font(22, True)
    d.text((55, 24), "GPT-2 Base: measured compression-quality operating points", font=title, fill="#102A43")
    left, top, right, bottom = 95, 105, 1120, 475
    d.line((left, bottom, right, bottom), fill="#60798C", width=3)
    d.line((left, top, left, bottom), fill="#60798C", width=3)
    for i, ratio in enumerate([1, 2, 3, 4]):
        y = bottom - int((ratio / 4.0) * (bottom - top))
        d.line((left, y, right, y), fill="#E0E6EB", width=2)
        d.text((45, y - 12), f"{ratio}x", font=axis, fill="#536575")
    labels = ["BF16\nbaseline", "4-bit +\nrank-64", "Adaptive 3/4-bit\n+ rank-32"]
    ratios = [1.0, 3.19, 3.51]
    ppls = ["58.5241 PPL", "58.7533 PPL\n(+0.39%)", "61.2229 PPL\n(+4.61%)"]
    fills = ["#A7B5C2", "#168C8C", "#1F5A85"]
    xs = [240, 580, 920]
    bar_w = 145
    for x, label_text, ratio, ppl, fill in zip(xs, labels, ratios, ppls, fills):
        h = int((ratio / 4.0) * (bottom - top))
        d.rounded_rectangle((x - bar_w / 2, bottom - h, x + bar_w / 2, bottom), radius=13, fill=fill)
        centered(d, (x, bottom - h - 28), f"{ratio:.2f}x", value, fill)
        line1, *rest = label_text.split("\n")
        centered(d, (x, bottom + 33), line1, font(20, True), "#263746")
        if rest:
            centered(d, (x, bottom + 59), rest[0], axis, "#536575")
        ppl_lines = ppl.split("\n")
        centered(d, (x, top + 18), ppl_lines[0], font(19, True), "#102A43")
        if len(ppl_lines) > 1:
            centered(d, (x, top + 46), ppl_lines[1], axis, "#536575")
    d.text((95, 545), "Bars show full-checkpoint compression relative to unique FP16 parameter payload.", font=axis, fill="#5B6773")
    path = ASSET_DIR / "gpt2_results.png"
    img.save(path, quality=95)
    return path


def make_tinyllama_results():
    img = Image.new("RGB", (1200, 590), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 24), "TinyLlama 1.1B: adaptive precision and real payload accounting", font=font(31, True), fill="#102A43")
    d.text((70, 92), "Layer allocation", font=font(23, True), fill="#1F5A85")
    x1, y1, x2, y2 = 75, 145, 565, 225
    total = 154
    segments = [(12, "#C28B20", "2-bit  12"), (31, "#168C8C", "3-bit  31"), (111, "#1F5A85", "4-bit  111")]
    cursor = x1
    for count, fill, label_text in segments:
        width = (x2 - x1) * count / total
        d.rectangle((cursor, y1, cursor + width, y2), fill=fill)
        cursor += width
    legend_y = 252
    for i, (_, fill, label_text) in enumerate(segments):
        lx = 82 + i * 165
        d.rounded_rectangle((lx, legend_y, lx + 26, legend_y + 26), radius=5, fill=fill)
        d.text((lx + 36, legend_y - 1), label_text, font=font(18, True), fill="#263746")

    d.text((680, 92), "Compression ratios", font=font(23, True), fill="#1F5A85")
    chart_left, chart_bottom = 690, 430
    for i, ratio in enumerate([1, 2, 3, 4]):
        y = chart_bottom - ratio * 72
        d.line((chart_left, y, 1120, y), fill="#E0E6EB", width=2)
        d.text((645, y - 10), f"{ratio}x", font=font(17), fill="#536575")
    vals = [2.70, 3.51]
    labs = ["Full file", "Matched matrices"]
    fills = ["#1F5A85", "#168C8C"]
    for i, (v, lab, fill) in enumerate(zip(vals, labs, fills)):
        x = 790 + i * 220
        h = v * 72
        d.rounded_rectangle((x - 55, chart_bottom - h, x + 55, chart_bottom), radius=12, fill=fill)
        centered(d, (x, chart_bottom - h - 24), f"{v:.2f}x", font(21, True), fill)
        centered(d, (x, chart_bottom + 30), lab, font(18, True), "#263746")

    rounded_box(d, (72, 330, 565, 490), "#F4F7FA", "#C9D3DC", radius=14, width=2)
    d.text((95, 352), "WikiText-2 (32 x 128 tokens)", font=font(20, True), fill="#102A43")
    d.text((95, 397), "BF16 baseline", font=font(19), fill="#536575")
    d.text((425, 397), "17.7996", font=font(21, True), fill="#102A43")
    d.text((95, 439), "Adaptive TMG-Q", font=font(19), fill="#536575")
    d.text((425, 439), "18.9206", font=font(21, True), fill="#168C8C")
    d.text((95, 515), "Full-file compression is lower because embeddings and metadata remain in the checkpoint.", font=font(18), fill="#5B6773")
    path = ASSET_DIR / "tinyllama_results.png"
    img.save(path, quality=95)
    return path


def make_recovery_chart():
    img = Image.new("RGB", (1200, 560), "white")
    d = ImageDraw.Draw(img)
    d.text((55, 24), "Calibration and residual recovery prevent sub-4-bit collapse", font=font(31, True), fill="#102A43")
    configs = [
        ("3-bit raw", 1142.3937, "#B8483B"),
        ("+ 0.1% residual", 94.7382, "#C28B20"),
        ("+ 1% residual", 71.5028, "#D6A743"),
        ("+ calibration", 18.1479, "#168C8C"),
        ("BF16 baseline", 13.5373, "#1F5A85"),
    ]
    left, right, top, bottom = 260, 1110, 100, 475
    max_log = math.log10(1200)
    for tick in [10, 30, 100, 300, 1000]:
        x = left + math.log10(tick) / max_log * (right - left)
        d.line((x, top, x, bottom), fill="#E0E6EB", width=2)
        d.text((x - 20, bottom + 18), str(tick), font=font(17), fill="#536575")
    for i, (name, ppl, fill) in enumerate(configs):
        y = top + 35 + i * 70
        width = math.log10(ppl) / max_log * (right - left)
        d.rounded_rectangle((left, y, left + width, y + 38), radius=9, fill=fill)
        d.text((40, y + 4), name, font=font(19, True), fill="#263746")
        d.text((left + width + 12, y + 4), f"{ppl:.4f}", font=font(18, True), fill=fill)
    d.text((880, 520), "Perplexity (log scale)", font=font(18, True), fill="#5B6773")
    path = ASSET_DIR / "recovery_chart.png"
    img.save(path, quality=95)
    return path


def make_assets():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "architecture": make_architecture(),
        "gpt2": make_gpt2_results(),
        "tinyllama": make_tinyllama_results(),
        "recovery": make_recovery_chart(),
    }


def header_footer(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.45)
    canvas.line(MARGIN_X, PAGE_H - 0.38 * inch, PAGE_W - MARGIN_X, PAGE_H - 0.38 * inch)
    canvas.setFont(BODY_FONT, 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_X, PAGE_H - 0.30 * inch, "TMG-Q | Technical Research Paper | June 2026")
    canvas.drawRightString(PAGE_W - MARGIN_X, 0.31 * inch, f"Page {page} of 5")
    canvas.restoreState()


class FivePageDocTemplate(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(
            filename,
            pagesize=letter,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title="TMG-Q: Calibration-Aware Mixed-Precision Quantization with Physical 2/3/4-bit Packing",
            author="Abdullah Salem Saleh Al-Faqeer",
            subject="Experimental post-training quantization for language models",
        )
        full = Frame(
            MARGIN_X,
            BOTTOM,
            PAGE_W - 2 * MARGIN_X,
            PAGE_H - TOP - BOTTOM,
            id="full",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        left = Frame(
            MARGIN_X,
            BOTTOM,
            COL_W,
            PAGE_H - TOP - BOTTOM,
            id="left",
            leftPadding=0,
            rightPadding=3,
            topPadding=0,
            bottomPadding=0,
        )
        right = Frame(
            MARGIN_X + COL_W + GUTTER,
            BOTTOM,
            COL_W,
            PAGE_H - TOP - BOTTOM,
            id="right",
            leftPadding=3,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="Full", frames=[full], onPage=header_footer),
                PageTemplate(id="Columns", frames=[left, right], onPage=header_footer),
            ]
        )


def P(text, style="PaperBody"):
    return Paragraph(text, styles[style])


def heading(number, title):
    return P(f"{number}. {title}", "Section")


def subheading(number, title):
    return P(f"{number} {title}", "Subsection")


def callout(text, fill=PALE_BLUE):
    table = Table([[P(text, "Callout")]], colWidths=[COL_W - 4])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), fill),
                ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def data_table(rows, widths, font_size=6.9):
    converted = []
    for r, row in enumerate(rows):
        converted.append(
            [
                Paragraph(
                    str(cell),
                    ParagraphStyle(
                        name=f"Cell{r}",
                        fontName=BOLD_FONT if r == 0 else BODY_FONT,
                        fontSize=font_size,
                        leading=font_size + 1.7,
                        textColor=WHITE if r == 0 else INK,
                        alignment=TA_LEFT if c == 0 else TA_CENTER,
                    ),
                )
                for c, cell in enumerate(row)
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for r in range(1, len(rows)):
        if r % 2 == 0:
            commands.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F5F8FA")))
    table.setStyle(TableStyle(commands))
    return table


def title_page(story, assets):
    story.append(Spacer(1, 0.18 * inch))
    story.append(
        Paragraph(
            "TMG-Q",
            ParagraphStyle(
                "TitleMark",
                fontName=BOLD_FONT,
                fontSize=26,
                leading=28,
                textColor=TEAL,
                alignment=TA_CENTER,
                spaceAfter=5,
            ),
        )
    )
    story.append(
        Paragraph(
            "Calibration-Aware Mixed-Precision Quantization<br/>with Physical 2/3/4-bit Weight Packing",
            ParagraphStyle(
                "PaperTitle",
                fontName=SERIF_BOLD,
                fontSize=19,
                leading=22,
                textColor=NAVY,
                alignment=TA_CENTER,
                spaceAfter=8,
            ),
        )
    )
    story.append(
        Paragraph(
            "Abdullah Salem Saleh Al-Faqeer<br/><font size='8.5' color='#5B6773'>Independent Researcher | June 2026</font>",
            ParagraphStyle(
                "Author",
                fontName=BODY_FONT,
                fontSize=10,
                leading=13,
                textColor=INK,
                alignment=TA_CENTER,
                spaceAfter=7,
            ),
        )
    )
    story.append(
        Paragraph(
            "github.com/abdallah2183/TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework",
            ParagraphStyle(
                "Repo",
                fontName=BODY_FONT,
                fontSize=7.5,
                leading=9,
                textColor=BLUE,
                alignment=TA_CENTER,
                spaceAfter=9,
            ),
        )
    )
    story.append(RLImage(str(assets["architecture"]), width=7.15 * inch, height=2.66 * inch))
    story.append(P("<b>Figure 1.</b> TMG-Q pipeline from model calibration to physical packing and measured validation.", "Caption"))
    story.append(
        Table(
            [
                [
                    P("<b>3.51x</b><br/><font size='7'>GPT-2 full-file compression</font>", "Callout"),
                    P("<b>+4.61%</b><br/><font size='7'>PPL change at that point</font>", "Callout"),
                    P("<b>2.70x</b><br/><font size='7'>TinyLlama full-file compression</font>", "Callout"),
                    P("<b>32 x 128</b><br/><font size='7'>WikiText-2 evaluation chunks</font>", "Callout"),
                ]
            ],
            colWidths=[1.78 * inch] * 4,
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), PALE_TEAL),
                    ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                    ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            ),
        )
    )
    story.append(Spacer(1, 6))
    story.append(P("<b>Abstract</b>", "Section"))
    story.append(
        P(
            "We present TMG-Q, an experimental post-training quantization framework for causal language models that combines physical INT32 bit packing, calibration-derived sensitivity estimates, learned group codebooks, sparse residual recovery, and adaptive mixed precision. The implementation supports 2-, 3-, and 4-bit weight storage for linear, GPT-2 Conv1D, embedding, and tied language-model-head modules. Unlike theoretical bit-count reports, TMG-Q measures both the serialized checkpoint and the payload of matched quantized matrices. Reproduced experiments on an NVIDIA RTX 5060 Ti show two practical GPT-2 Base operating points: 3.19x full-file compression with a 0.39% perplexity increase, and 3.51x compression with a 4.61% increase. On TinyLlama 1.1B, adaptive 2/3/4-bit allocation reaches 2.70x full-file compression and 3.51x compression over matched matrices, with perplexity increasing from 17.7996 to 18.9206. The results establish a reproducible research baseline while showing that robust 2-bit operation and optimized packed inference remain open problems.",
            "Abstract",
        )
    )
    story.append(
        P(
            "<b>Keywords:</b> large language models, post-training quantization, mixed precision, codebook quantization, model compression, perplexity.",
            "Keyword",
        )
    )
    story.append(NextPageTemplate("Columns"))
    story.append(PageBreak())


def page_two(story):
    story.append(heading("1", "Introduction"))
    story.append(
        P(
            "Autoregressive language models are expensive to store and move through memory because every generated token requires repeated access to large parameter tensors. Weight-only post-training quantization (PTQ) addresses this bottleneck by replacing floating-point weights with low-bit representations while retaining higher-precision activations. The practical objective is not simply to minimize reconstruction error: a usable method must jointly manage model quality, checkpoint bytes, calibration cost, architectural compatibility, and runtime support.",
        )
    )
    story.append(
        P(
            "TMG-Q was developed as an end-to-end laboratory for this trade-off. Its modern path quantizes Hugging Face causal language models on CUDA, emits portable <font name='Courier'>.pt</font> checkpoints, reloads the packed modules for evaluation, and reports actual serialized size. This paper focuses only on results reproduced with that path. Earlier project observations for GPT-2 Medium, GPT-2 Large, NanoGPT, and LLaVA are excluded from the primary evidence because they were produced by different scripts or evaluation protocols.",
        )
    )
    story.append(subheading("1.1", "Contributions"))
    story.append(
        P(
            "<b>Physical low-bit storage.</b> A common failure in experimental quantization is to retain integer values in byte-addressable tensors while claiming 2- or 3-bit storage. TMG-Q packs indices into INT32 containers: 16 values at 2-bit, 10 at 3-bit, and 8 at 4-bit.",
        )
    )
    story.append(
        P(
            "<b>Adaptive precision selection.</b> Each projection is probed under 2-bit codebook, 3-bit codebook, and 4-bit linear candidates. The lowest precision satisfying a Hessian-weighted normalized error threshold is selected.",
        )
    )
    story.append(
        P(
            "<b>Recovery mechanisms.</b> The framework supports row-wise clipping, sparse high-precision outlier residuals, optional low-rank SVD residuals, and low-rank vocabulary logit distillation.",
        )
    )
    story.append(
        P(
            "<b>Reproducible accounting.</b> Evaluation reports perplexity, complete checkpoint ratio, matched-matrix ratio, bit allocation, and explicit rejection when a configured quality gate is exceeded.",
        )
    )
    story.append(callout("<b>Scope.</b> The present evidence supports checkpoint compression, not inference acceleration. Packed weights are dequantized by PyTorch modules; a fused low-bit CUDA matrix-multiplication kernel has not yet been implemented.", PALE_GOLD))

    story.append(heading("2", "Related Work"))
    story.append(
        P(
            "GPTQ introduced a scalable one-shot method based on approximate second-order information and demonstrated accurate 3- and 4-bit weight quantization [1]. AWQ observed that activation magnitude reveals salient channels and protects them through equivalent scaling, pairing the method with optimized 4-bit inference kernels [2]. TMG-Q shares the use of calibration statistics but differs in selecting among heterogeneous 2/3/4-bit candidate encodings and in exposing full serialized accounting.",
        )
    )
    story.append(
        P(
            "Extreme compression methods increasingly use richer representations. AQLM applies learned additive codebooks and block-level optimization below 3 bits per parameter [3]. QuIP# combines randomized Hadamard incoherence processing with lattice codebooks and fine-tuning [4]. These works indicate why uniform scalar 2-bit rounding is insufficient, a result also observed directly in TMG-Q.",
        )
    )
    story.append(
        P(
            "Sparse recovery is another established direction. SpQR isolates high-impact outliers in higher precision [5], while SqueezeLLM combines sensitivity-based non-uniform quantization with a dense-and-sparse decomposition [6]. TMG-Q's sparse residual mechanism follows the same broad principle, but its current implementation remains a research prototype rather than an optimized inference format.",
        )
    )
    story.append(
        data_table(
            [
                ["Method", "Primary mechanism", "Typical regime"],
                ["GPTQ", "Approx. second-order updates", "3-4 bit"],
                ["AWQ", "Activation-aware channel scaling", "4 bit"],
                ["AQLM", "Additive learned codebooks", "2-3 bit"],
                ["QuIP#", "Incoherence + lattice codebooks", "2-4 bit"],
                ["SpQR / SqueezeLLM", "Sparse outlier isolation", "3-4 bit"],
                ["TMG-Q", "Adaptive codebook/linear + residuals", "2/3/4 bit"],
            ],
            [0.85 * inch, 1.35 * inch, 0.75 * inch],
            6.45,
        )
    )
    story.append(P("<b>Table 1.</b> Positioning of TMG-Q relative to representative weight-only PTQ methods.", "Caption"))
    story.append(subheading("2.1", "Research Gap and Design Goals"))
    story.append(
        P(
            "Published systems demonstrate strong quality, but reproducing their full stack can require specialized kernels, block reconstruction, or vector codebooks. TMG-Q asks a narrower engineering question: how far can a transparent PyTorch research pipeline push real checkpoint compression while preserving auditable byte accounting and a direct path from export to perplexity evaluation?",
        )
    )
    story.append(
        P(
            "The design therefore prioritizes four properties: <b>(i)</b> every claimed low-bit index is physically packed; <b>(ii)</b> layer choices are derived from calibration rather than fixed position rules; <b>(iii)</b> failed configurations remain visible in the experiment record; and <b>(iv)</b> a compressed checkpoint can be reloaded without the original floating-point weights.",
        )
    )
    story.append(
        callout(
            "<b>Research questions.</b> RQ1: Can adaptive 2/3/4-bit allocation improve the compression-quality frontier over fixed policies? RQ2: Which recovery mechanisms are necessary below 4-bit? RQ3: How much does complete checkpoint accounting differ from matrix-only accounting?",
            PALE_TEAL,
        )
    )
    story.append(PageBreak())


def page_three(story, assets):
    story.append(heading("3", "Method"))
    story.append(subheading("3.1", "Grouped Quantization and Packing"))
    story.append(
        P(
            "For a weight row divided into groups of size <i>g</i>, the linear path estimates a scale and offset after clipping search. A weight <i>w</i> is mapped to an integer index <i>q</i> in the range [0, 2<super>b</super>-1]. The codebook path instead learns 2<super>b</super> centroids for each group by Lloyd-style updates. At load time, indices select centroids or reconstruct affine values.",
        )
    )
    story.append(
        callout(
            "<b>Linear reconstruction:</b> q = clip(round(w / s) + z), and w-hat = (q - z)s.<br/><b>Codebook reconstruction:</b> w-hat = C[q], where C stores group-specific learned centroids.",
            PALE_BLUE,
        )
    )
    story.append(
        P(
            "The integer indices are packed with bitwise shifts into INT32 tensors. Metadata, codebooks, residuals, shape descriptors, and alignment waste are included in the measured payload. Three-bit storage uses 30 of every 32 container bits, so its practical size is slightly above an idealized three-bit stream.",
        )
    )
    story.append(
        data_table(
            [
                ["Bits", "Values / INT32", "Index utilization", "Candidate"],
                ["2", "16", "32 / 32", "Learned codebook"],
                ["3", "10", "30 / 32", "Learned codebook"],
                ["4", "8", "32 / 32", "Affine linear"],
            ],
            [0.48 * inch, 0.78 * inch, 0.88 * inch, 0.82 * inch],
            6.6,
        )
    )
    story.append(P("<b>Table 2.</b> Physical index packing used in the modern TMG-Q checkpoint format.", "Caption"))
    story.append(subheading("3.2", "Calibration and Sensitivity"))
    story.append(
        P(
            "Calibration forwards WikiText-2 token blocks through the uncompressed model and accumulates input-channel energy for each projection. These statistics act as a diagonal Hessian proxy. Candidate reconstruction error is weighted by this proxy so that errors on frequently excited channels receive greater penalty.",
        )
    )
    story.append(
        callout(
            "<b>Selection metric:</b> NMSE<sub>H</sub> = sum(H * (W - W-hat)<super>2</super>) / max(sum(H * W<super>2</super>), epsilon).",
            PALE_TEAL,
        )
    )
    story.append(subheading("3.3", "Adaptive and Budget Policies"))
    story.append(
        P(
            "The adaptive policy tests 2-bit codebook, 3-bit codebook, and 4-bit linear candidates on representative rows. It accepts the first candidate below its threshold; otherwise it chooses the lowest-error candidate. A minimum-bit constraint can disable unstable regimes. The budget policy estimates candidate payloads for all layers and optimizes a global assignment under a target ratio, rejecting mathematically infeasible requests before full export.",
        )
    )
    story.append(subheading("3.4", "Residual and Vocabulary Recovery"))
    story.append(
        P(
            "A sparse residual stores a configurable fraction of the largest reconstruction errors as row, column, and value triples. Optional SVD factors approximate structured residual error, although the tested TinyLlama configurations did not justify their additional bytes. For GPT-2, quantizing the tied vocabulary directly caused severe degradation. TMG-Q therefore freezes all packed transformer weights and trains a low-rank correction against teacher logits. This localized distillation recovered vocabulary quality without restoring the full embedding matrix.",
        )
    )
    story.append(RLImage(str(assets["recovery"]), width=3.0 * inch, height=1.4 * inch))
    story.append(P("<b>Figure 2.</b> Quick-sweep TinyLlama evidence that calibration and sparse recovery are essential at 3-bit.", "Caption"))
    story.append(subheading("3.5", "Export Algorithm"))
    story.append(
        data_table(
            [
                ["Step", "Operation"],
                ["1", "Load BF16 model and tokenizer; enumerate quantizable modules."],
                ["2", "Run calibration blocks and accumulate per-channel energy."],
                ["3", "Probe low-bit candidates on representative rows."],
                ["4", "Select precision adaptively or solve the global payload budget."],
                ["5", "Quantize the full tensor; attach sparse/SVD recovery if enabled."],
                ["6", "Pack indices, save metadata, reload, evaluate, and measure bytes."],
            ],
            [0.38 * inch, 2.54 * inch],
            6.25,
        )
    )
    story.append(P("<b>Algorithm 1.</b> High-level TMG-Q export and verification sequence.", "Caption"))
    story.append(subheading("3.6", "Complexity and Implementation"))
    story.append(
        P(
            "Quantization proceeds one module at a time, limiting peak device memory to the model plus temporary candidate tensors. Adaptive probing reduces candidate cost by evaluating sampled rows before one complete quantization pass. Codebook updates scale with the number of groups, centroids, iterations, and probed elements; linear clipping search is cheaper. The exported state dictionary stores only packed modules, required metadata, and any selected residual parameters.",
        )
    )
    story.append(
        P(
            "The loader reconstructs quantized module classes dynamically from shape and bit-width metadata. This architectural replacement is important for GPT-2 because its projections use the transposed Conv1D convention rather than standard PyTorch Linear storage.",
        )
    )
    story.append(PageBreak())


def page_four(story, assets):
    story.append(heading("4", "Experimental Protocol"))
    story.append(
        P(
            "Experiments were executed locally on an NVIDIA GeForce RTX 5060 Ti with 16 GB VRAM using a CUDA-enabled PyTorch 2.12 development build and Hugging Face Transformers. GPT-2 Base and TinyLlama-1.1B-Chat-v1.0 were evaluated on the WikiText-2 test split [7]. The reported verification protocol uses 32 non-overlapping chunks of 128 tokens in BF16 runtime. Calibration uses separate training-split chunks.",
        )
    )
    story.append(
        P(
            "Perplexity is computed by teacher-forced causal language-model loss. Complete checkpoint compression compares the serialized TMG-Q file with the model's unique FP16 parameter payload. Matched-matrix compression compares only tensors replaced by quantized modules. This distinction prevents embeddings, preserved parameters, and metadata from being hidden behind a theoretical bit count.",
        )
    )
    story.append(
        data_table(
            [
                ["Item", "GPT-2 Base", "TinyLlama 1.1B"],
                ["Evaluation", "32 x 128 tokens", "32 x 128 tokens"],
                ["Runtime dtype", "BF16", "BF16"],
                ["Calibration", "WikiText-2 train", "WikiText-2 train"],
                ["Quantized modules", "49", "154"],
                ["Hardware", "RTX 5060 Ti 16 GB", "RTX 5060 Ti 16 GB"],
            ],
            [0.9 * inch, 1.02 * inch, 1.02 * inch],
            6.45,
        )
    )
    story.append(P("<b>Table 3.</b> Reproduced evaluation environment and scope.", "Caption"))
    story.append(heading("5", "Results"))
    story.append(subheading("5.1", "GPT-2 Base"))
    story.append(RLImage(str(assets["gpt2"]), width=3.03 * inch, height=1.49 * inch))
    story.append(P("<b>Figure 3.</b> Full-file compression and perplexity for the verified GPT-2 operating points.", "Caption"))
    story.append(
        P(
            "The quality-oriented checkpoint uses 4-bit packed weights and a rank-64 distilled vocabulary. It reduces the complete file from approximately 237.4 MiB to 74.5 MiB (3.19x) while increasing perplexity from 58.5241 to 58.7533, a relative change of 0.39%. The compression-oriented checkpoint uses two 3-bit and 47 4-bit matrices plus a rank-32 vocabulary correction. Its 67.7 MiB file reaches 3.51x compression with perplexity 61.2229, a 4.61% increase.",
        )
    )
    story.append(
        data_table(
            [
                ["Operating point", "File", "Ratio", "PPL", "Relative change"],
                ["BF16 baseline", "237.4 MiB", "1.00x", "58.5241", "-"],
                ["4-bit + rank-64", "74.5 MiB", "3.19x", "58.7533", "+0.39%"],
                ["Adaptive 3/4-bit + rank-32", "67.7 MiB", "3.51x", "61.2229", "+4.61%"],
            ],
            [1.25 * inch, 0.58 * inch, 0.45 * inch, 0.52 * inch, 0.65 * inch],
            6.1,
        )
    )
    story.append(P("<b>Table 4.</b> GPT-2 Base results reproduced on June 7, 2026.", "Caption"))
    story.append(subheading("5.2", "TinyLlama 1.1B"))
    story.append(RLImage(str(assets["tinyllama"]), width=3.03 * inch, height=1.49 * inch))
    story.append(P("<b>Figure 4.</b> TinyLlama layer allocation, compression ratios, and verified perplexity.", "Caption"))
    story.append(
        P(
            "Adaptive TinyLlama quantization assigns 12 projections to 2-bit codebooks, 31 to 3-bit codebooks, and 111 to 4-bit linear storage. With group size 64, calibration, and a 0.1% sparse residual, perplexity rises from 17.7996 to 18.9206 (+6.30%). The matched quantized matrices compress by 3.51x, while the complete checkpoint compresses by 2.70x because embeddings, preserved parameters, and metadata remain.",
        )
    )
    story.append(
        data_table(
            [
                ["Metric", "BF16", "Adaptive TMG-Q"],
                ["WikiText-2 PPL", "17.7996", "18.9206"],
                ["Relative PPL change", "-", "+6.30%"],
                ["Full payload", "2098.2 MiB", "778.0 MiB"],
                ["Full-file ratio", "1.00x", "2.70x"],
                ["Matched matrices", "1848.0 MiB", "526.9 MiB"],
                ["Matched-matrix ratio", "1.00x", "3.51x"],
            ],
            [1.18 * inch, 0.75 * inch, 1.0 * inch],
            6.25,
        )
    )
    story.append(P("<b>Table 5.</b> TinyLlama full-file and matrix-level accounting.", "Caption"))
    story.append(
        callout(
            "<b>Key observation.</b> Reporting only the 3.51x matrix ratio would overstate deployable compression. The complete checkpoint ratio is 2.70x after preserved tensors and metadata are counted.",
            PALE_GOLD,
        )
    )
    story.append(PageBreak())


def page_five(story):
    story.append(heading("6", "Analysis and Discussion"))
    story.append(subheading("6.1", "What Worked"))
    story.append(
        P(
            "The strongest result is not a single quantizer but a coordinated pipeline. Calibration consistently improved TinyLlama 4-bit quality and transformed the 3-bit linear path from collapse into a usable regime when combined with residual recovery. Learned codebooks were substantially better than linear quantization at 2 and 3 bits. Adaptive allocation then exploited easy layers without forcing the entire model into an unstable precision.",
        )
    )
    story.append(
        P(
            "Vocabulary distillation was decisive for GPT-2. A fully tied 4-bit vocabulary without training produced perplexity above 1700 in the initial test, whereas a rank-64 correction returned the 32-chunk result to 58.7533. This indicates that embedding and output-head error can dominate checkpoint quality even when transformer projections are well reconstructed.",
        )
    )
    story.append(subheading("6.2", "Negative Results"))
    story.append(
        P(
            "Uniform 2-bit linear quantization remains unusable for TinyLlama. A 1% residual and rank-8 SVD correction reduced damage but did not restore acceptable perplexity. Metadata compression was also fragile: experimental FP8 conversion of linear scales and offsets caused catastrophic error. QAT-lite reduced calibration divergence for selected codebooks but did not improve held-out WikiText-2 perplexity in the tested configurations.",
        )
    )
    story.append(
        data_table(
            [
                ["Experiment", "Result", "Decision"],
                ["3-bit raw, g128", "PPL 1142.3937", "Rejected"],
                ["3-bit + 1% residual + calibration", "PPL 18.1479", "Recovered"],
                ["2-bit + 1% residual + rank-8 SVD", "PPL 431.8017", "Rejected"],
                ["GPT-2 rank-24 vocabulary", "+7.91% PPL", "Rejected"],
                ["FP8 vocabulary residual", "+5.02% PPL", "Outside gate"],
                ["FP8 linear scales / zeros", "PPL 4622.9760", "Rejected"],
            ],
            [1.25 * inch, 1.0 * inch, 0.72 * inch],
            6.25,
        )
    )
    story.append(P("<b>Table 6.</b> Representative negative results retained to make the research boundary explicit.", "Caption"))
    story.append(subheading("6.3", "Limitations and Threats to Validity"))
    story.append(
        P(
            "The verification subset contains 4096 evaluated tokens per model and is therefore smaller than standardized full-corpus protocols. Results have not been independently reproduced. The two model families differ in architecture and tokenizer, so their perplexities cannot be compared directly. Checkpoint compression does not imply latency or energy improvement because no fused packed GEMM kernel is present. Finally, thresholds and calibration choices were tuned through local sweeps, creating a risk of selection bias.",
        )
    )
    story.append(subheading("6.4", "Future Work"))
    story.append(
        P(
            "The next evaluation milestone is integration with lm-evaluation-harness and a standardized WikiText-2/C4 protocol on the same model families used by published baselines. Methodologically, randomized rotations or other incoherence processing should be evaluated before further claims at 2-bit. Additive or vector codebooks may also improve the accuracy-per-byte frontier beyond scalar group codebooks.",
        )
    )
    story.append(
        P(
            "On the systems side, fused dequantization-GEMM CUDA kernels are required to test whether checkpoint savings translate into decoding speed. Safetensors or GGUF serialization, deterministic calibration manifests, and public checkpoint hashes would strengthen external reproducibility.",
        )
    )
    story.append(heading("7", "Conclusion"))
    story.append(
        P(
            "TMG-Q demonstrates that a transparent combination of physical packing, calibration-aware sensitivity, learned codebooks, sparse recovery, and localized distillation can produce meaningful low-bit compression on real language models. The best verified full-file result is 3.51x on GPT-2 Base within a 5% perplexity gate; TinyLlama reaches 2.70x full-file and 3.51x matched-matrix compression with a 6.30% perplexity increase. The next research priorities are standardized evaluation, rotation or incoherence processing for robust 2-bit quantization, and fused CUDA kernels that convert storage gains into measured inference speedups.",
        )
    )
    story.append(P("References", "Section"))
    references = [
        "[1] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,” ICLR, 2023. arXiv:2210.17323.",
        "[2] J. Lin et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration,” MLSys, 2024. arXiv:2306.00978.",
        "[3] V. Egiazarian et al., “Extreme Compression of Large Language Models via Additive Quantization,” ICML, 2024. arXiv:2401.06118.",
        "[4] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, and C. De Sa, “QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks,” ICML, 2024. arXiv:2402.04396.",
        "[5] T. Dettmers et al., “SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression,” ICLR, 2024. arXiv:2306.03078.",
        "[6] S. Kim et al., “SqueezeLLM: Dense-and-Sparse Quantization,” ICML, 2024. arXiv:2306.07629.",
        "[7] S. Merity, C. Xiong, J. Bradbury, and R. Socher, “Pointer Sentinel Mixture Models,” ICLR, 2017. arXiv:1609.07843.",
    ]
    for ref in references:
        story.append(P(ref, "Reference"))
    story.append(
        callout(
            "<b>Reproducibility artifact.</b> Source code, evaluation utilities, experiment report, and checkpoint accounting are available in the public TMG-Q repository.",
            PALE_TEAL,
        )
    )


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assets = make_assets()
    doc = FivePageDocTemplate(str(PDF_PATH))
    story = []
    title_page(story, assets)
    page_two(story)
    page_three(story, assets)
    page_four(story, assets)
    page_five(story)
    doc.build(story)
    print(PDF_PATH)


if __name__ == "__main__":
    build()

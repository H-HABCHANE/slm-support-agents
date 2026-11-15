"""Génère les fichiers multimodaux du projet :
- data/kb/terms_of_sale.pdf : conditions générales (document PDF de la base de connaissances) ;
- data/samples/ : tickets d'exemple en texte, en PDF (courrier scanné) et en image (capture d'écran).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / "data" / "kb"
SAMPLES = ROOT / "data" / "samples"

TERMS = [
    ("NovaShop – General Terms of Sale (extract)", None),
    ("Article 4 – Right of withdrawal",
     "Under EU consumer law, customers have a right of withdrawal of 14 days from delivery, without having to "
     "give a reason. NovaShop extends this period to 30 days for most items (see the refund policy). The right "
     "of withdrawal does not apply to custom-made products."),
    ("Article 6 – Legal warranty",
     "All products benefit from the 2-year legal warranty of conformity from the delivery date. A defective "
     "product is repaired or replaced free of charge; if neither is possible, the customer is refunded."),
    ("Article 7 – Lost or damaged parcels",
     "A parcel that arrives damaged must be refused or reported within 48 hours of delivery with photos. "
     "NovaShop then ships a replacement or issues a full refund, including shipping costs."),
    ("Article 9 – Personal data",
     "Personal data is processed in accordance with the GDPR and stored in the European Union. Customers can "
     "request access, correction or deletion of their data at privacy@novashop.example. Invoices are kept for "
     "10 years to meet accounting obligations."),
]

TICKET_TXT = (
    "Hello,\n\nI ordered a blender last Monday (order NS-482913) and the status still says Preparing. "
    "I chose Express delivery because I need it for this weekend. When will it arrive?\n\nThanks, Julie"
)

TICKET_PDF = [
    "Paris, 3 November 2026",
    "Subject: refund not received",
    "Dear NovaShop team,",
    "I returned a pair of headphones (order NS-118204) three weeks ago. The tracking shows the warehouse "
    "received the parcel on 20 October, but I still have not received my refund on my Visa card. "
    "Could you tell me where my refund is?",
    "Best regards, Marc Dupont",
]

SCREENSHOT_LINES = [
    "NovaShop - Checkout",
    "",
    "Payment declined",
    "Your card was declined. Please try again",
    "or choose another payment method.",
    "",
    "Customer note: I tried 3 times with my Mastercard,",
    "I was charged twice but the order failed!",
]


def make_terms_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for title, body in TERMS:
        story.append(Paragraph(title, styles["Title" if body is None else "Heading2"]))
        if body:
            story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 8))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def make_ticket_pdf(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for line in TICKET_PDF:
        story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 10))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def make_screenshot(path: Path) -> None:
    img = Image.new("RGB", (900, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 900, 60], fill=(25, 90, 170))
    y = 15
    for i, line in enumerate(SCREENSHOT_LINES):
        color = "white" if i == 0 else ((200, 30, 30) if line.startswith("Payment") else "black")
        draw.text((30, y), line, fill=color, font=_font(30 if i == 0 else 26))
        y += 48 if i == 0 else 40
    img.save(path)


def main() -> None:
    KB.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    make_terms_pdf(KB / "terms_of_sale.pdf")
    (SAMPLES / "ticket_email.txt").write_text(TICKET_TXT, encoding="utf-8")
    make_ticket_pdf(SAMPLES / "ticket_letter.pdf")
    make_screenshot(SAMPLES / "ticket_screenshot.png")
    print("Fichiers générés dans", KB, "et", SAMPLES)


if __name__ == "__main__":
    main()

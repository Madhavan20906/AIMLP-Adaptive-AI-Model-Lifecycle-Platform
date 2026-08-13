"""
deployment/report_generator.py

Generates report.pdf from a Mode 1 (or Mode 2) result -- the piece named
in the original spec's output list that hadn't been built yet. Covers:
dataset profile, candidate selection reasoning, full leaderboard, and the
winning model's metrics -- everything needed to explain WHY a model was
chosen, not just which one.
"""

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

ACCENT = colors.HexColor("#2563eb")
MUTED = colors.HexColor("#6b7280")
LIGHT_BG = colors.HexColor("#f3f4f6")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportTitle", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#111827"),
    ))
    styles.add(ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=14, spaceBefore=18, spaceAfter=8,
        textColor=colors.HexColor("#111827"),
    ))
    styles.add(ParagraphStyle(
        "Meta", parent=styles["Normal"], fontSize=9, textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        "Reasoning", parent=styles["Normal"], fontSize=9.5, textColor=colors.HexColor("#374151"),
        leftIndent=10, spaceAfter=4,
    ))
    return styles


def _profile_table(profile: dict, styles):
    rows = [
        ["Rows", f"{profile['n_rows']:,}", "Columns", str(profile["n_columns"])],
        ["Problem type", profile["problem_type"], "Target column", profile["target_column"]],
        ["Data quality score", f"{profile['data_quality_score']} / 100", "Missing ratio", f"{profile['missing_ratio']*100:.1f}%"],
        ["Duplicates", str(profile["duplicates"]), "Imbalance ratio", str(profile.get("imbalance_ratio") or "—")],
    ]
    t = Table(rows, colWidths=[110, 150, 110, 150])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), MUTED),
        ("TEXTCOLOR", (2, 0), (2, -1), MUTED),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#e5e7eb")),
    ]))
    return t


def _leaderboard_table(leaderboard: list, problem_type: str):
    if problem_type == "classification":
        header = ["Rank", "Algorithm", "Score", "Accuracy", "F1", "ROC AUC", "Train (s)", "Size (KB)"]
        rows = [header]
        for r in leaderboard:
            m = r["metrics"]
            rows.append([
                f"#{r.get('rank', '-')}", r["name"], str(r["overall_score"]),
                f"{m.get('accuracy', 0):.3f}", f"{m.get('f1', 0):.3f}",
                f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "—",
                str(r["train_time_s"]), str(r["model_size_kb"]),
            ])
    else:
        header = ["Rank", "Algorithm", "Score", "R2", "RMSE", "Train (s)", "Size (KB)"]
        rows = [header]
        for r in leaderboard:
            m = r["metrics"]
            rows.append([
                f"#{r.get('rank', '-')}", r["name"], str(r["overall_score"]),
                f"{m.get('r2', 0):.3f}", f"{m.get('rmse', 0):.3f}",
                str(r["train_time_s"]), str(r["model_size_kb"]),
            ])

    t = Table(rows, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TEXTCOLOR", (2, 1), (2, -1), ACCENT),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e5e7eb")),
    ]
    t.setStyle(TableStyle(style))
    return t


def generate_mode1_report(
    project_name: str,
    profile: dict,
    candidate_selection: dict,
    leaderboard: list,
    best_model: dict,
    total_time_s: float,
    output_path: str = None,
) -> Path:
    styles = _styles()
    output_path = Path(output_path or f"./deployment_output/{project_name}_report.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    story = []

    story.append(Paragraph("AIMLP Training Report", styles["ReportTitle"]))
    story.append(Paragraph(
        f"Project: <b>{project_name}</b> &nbsp;&middot;&nbsp; "
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"&nbsp;&middot;&nbsp; Total training time: {total_time_s}s",
        styles["Meta"],
    ))

    story.append(Paragraph("Dataset Analysis", styles["SectionHeading"]))
    story.append(_profile_table(profile, styles))

    story.append(Paragraph("Candidate Algorithm Selection", styles["SectionHeading"]))
    confidence_pct = round(candidate_selection["confidence"] * 100)
    mode_desc = (
        "evaluated the full model pool (confidence too low to shortlist)"
        if candidate_selection["evaluated_all"]
        else f"shortlisted with {confidence_pct}% confidence"
    )
    story.append(Paragraph(
        f"<b>{mode_desc}</b>: {', '.join(candidate_selection['candidates_evaluated'])}",
        styles["Normal"],
    ))
    story.append(Spacer(1, 6))
    for reason in candidate_selection["reasons"]:
        story.append(Paragraph(f"&bull; {reason}", styles["Reasoning"]))

    story.append(Paragraph("Leaderboard", styles["SectionHeading"]))
    story.append(_leaderboard_table(leaderboard, profile["problem_type"]))

    story.append(Paragraph("Recommendation", styles["SectionHeading"]))
    story.append(Paragraph(
        f"<b>{best_model['name']}</b> was selected as the best-performing model with an overall "
        f"score of <b>{best_model['overall_score']}</b>, evaluated across "
        f"{len(leaderboard)} candidate algorithms. See the metrics above for the full "
        f"breakdown (accuracy, F1, ROC AUC, training time, and model size) that this "
        f"recommendation is based on.",
        styles["Normal"],
    ))

    doc.build(story)
    return output_path

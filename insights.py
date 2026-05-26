import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF

REPORTS_DIR = 'outputs/reports/'
CHARTS_DIR  = 'outputs/charts/'
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── 1. Console insights ──────────────────────────────────────────────────────
def print_insights(df, model, X, acc):
    import numpy as np

    print("\n" + "="*55)
    print("       AIRLINE SATISFACTION – KEY INSIGHTS")
    print("="*55)

    # Top 3 features
    feat = pd.Series(model.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
    print("\n📌 Top 3 Drivers of Satisfaction:")
    for i, (name, val) in enumerate(feat.head(3).items(), 1):
        print(f"   {i}. {name:35s} → {val*100:.1f}%")

    # Satisfaction rate by class
    class_map = {0: 'Business', 1: 'Eco', 2: 'Eco Plus'}
    sat_by_class = df.groupby('Class')['satisfaction'].mean()
    print("\n📌 Satisfaction Rate by Class:")
    for cls, rate in sat_by_class.items():
        label = class_map.get(cls, str(cls))
        print(f"   {label:12s} → {rate*100:.1f}%")

    # Satisfaction rate by travel type
    travel_map = {0: 'Business Travel', 1: 'Personal Travel'}
    sat_by_travel = df.groupby('Type of Travel')['satisfaction'].mean()
    print("\n📌 Satisfaction Rate by Travel Type:")
    for t, rate in sat_by_travel.items():
        label = travel_map.get(t, str(t))
        print(f"   {label:20s} → {rate*100:.1f}%")

    # Average delay for satisfied vs dissatisfied
    avg_delay = df.groupby('satisfaction')['Departure Delay in Minutes'].mean()
    print("\n📌 Avg Departure Delay (minutes):")
    print(f"   Dissatisfied → {avg_delay.get(0, 0):.1f} min")
    print(f"   Satisfied    → {avg_delay.get(1, 0):.1f} min")

    print(f"\n📌 Overall Model Accuracy: {acc*100:.2f}%")
    print("="*55)


# ── 2. Insights bar chart ────────────────────────────────────────────────────
def plot_insights_chart(df, model, X):
    feat = pd.Series(model.feature_importances_,
                     index=X.columns).sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(feat.index[::-1], feat.values[::-1],
                    color='steelblue', edgecolor='white')
    for bar, val in zip(bars, feat.values[::-1]):
        plt.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val*100:.1f}%', va='center', fontsize=9)
    plt.title('Top 10 Features Impacting Passenger Satisfaction',
              fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    path = CHARTS_DIR + 'insights_top_features.png'
    plt.savefig(path, dpi=150)
    plt.show()
    print(f"✅ Insights chart saved → {path}")


# ── 3. Recommendations ───────────────────────────────────────────────────────
RECOMMENDATIONS = [
    ("Online Boarding",
     "Streamline the app/web check-in flow; add real-time gate notifications."),
    ("Inflight Entertainment",
     "Expand content library; ensure screens work on every seat."),
    ("Seat Comfort",
     "Upgrade Economy seating; increase legroom on long-haul routes."),
    ("Inflight Wifi",
     "Provide free or low-cost high-speed Wi-Fi, especially for Business."),
    ("On-board Service",
     "Invest in crew training for personalised, proactive service."),
    ("Departure Delays",
     "Improve ground operations; notify passengers early via SMS/app."),
]

def print_recommendations():
    print("\n" + "="*55)
    print("       RECOMMENDATIONS")
    print("="*55)
    for i, (area, tip) in enumerate(RECOMMENDATIONS, 1):
        print(f"\n  {i}. {area}")
        print(f"     → {tip}")
    print("="*55)


# ── 4. PDF Report ────────────────────────────────────────────────────────────
def generate_pdf_report(acc):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 12, 'Airline Customer Experience – Insights Report', ln=True)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Model Accuracy: {acc*100:.2f}%', ln=True)
    pdf.ln(4)

    # Section: Key Findings
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, 'Key Findings', ln=True)
    pdf.set_font('Arial', '', 11)
    findings = [
        "Online Boarding is the strongest predictor of satisfaction.",
        "Business class passengers are more satisfied than Economy.",
        "Business travelers show higher satisfaction than personal travelers.",
        "Long departure delays strongly correlate with dissatisfaction.",
        "Inflight entertainment and seat comfort are critical service factors.",
    ]
    for f in findings:
        pdf.multi_cell(0, 8, f'  • {f}')
    pdf.ln(4)

    # Section: Recommendations
    pdf.set_font('Arial', 'B', 13)
    pdf.cell(0, 10, 'Recommendations', ln=True)
    pdf.set_font('Arial', '', 11)
    for area, tip in RECOMMENDATIONS:
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, f'  {area}', ln=True)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 7, f'    {tip}')
    pdf.ln(4)

    # Embed charts
    for img_name, caption in [
        ('insights_top_features.png', 'Top 10 Feature Importance'),
        ('confusion_matrix.png',      'Confusion Matrix'),
        ('satisfaction_distribution.png', 'Satisfaction Distribution'),
    ]:
        img_path = CHARTS_DIR + img_name
        if os.path.exists(img_path):
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 10, caption, ln=True)
            pdf.image(img_path, x=15, w=180)

    report_path = REPORTS_DIR + 'airline_insights_report.pdf'
    pdf.output(report_path)
    print(f"\n✅ PDF report saved → {report_path}")


# ── 5. Run all insights standalone ───────────────────────────────────────────
def run_all_insights(df, model, X, acc):
    print_insights(df, model, X, acc)
    plot_insights_chart(df, model, X)
    print_recommendations()
    generate_pdf_report(acc)
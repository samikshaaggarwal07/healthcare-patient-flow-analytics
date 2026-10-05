# 🏥 Healthcare Patient Flow Analytics

> **Educational Project** — This dashboard was built for learning and portfolio purposes using a synthetic/public dataset. It does not represent real patient data and must not be used for clinical decisions.

---

## 📌 Project Description

A complete end-to-end Healthcare Patient Flow Analytics solution that cleans raw hospital data, performs exploratory analysis, experiments with a predictive model, and presents findings in a multi-page interactive Streamlit dashboard. The project follows the **Data → Insights → Decisions → Actions** flow, designed for a non-technical hospital manager audience.

---

## 🩺 Problem Statement

A hospital wants to understand patient flow:
- How many patients arrive, and when (by hour, weekday, month)?
- How long do patients wait, and which departments have the worst wait times?
- Which departments receive the most referrals?
- How satisfied are patients, and does wait time affect satisfaction?
- What should management do to reduce wait times and improve patient experience?

---

## 📂 Dataset

The dataset contains 9,216 synthetic emergency department visits from **April 2023 to October 2024**.

**Download:** [healthcare_data.csv on Google Drive](https://drive.google.com/file/d/1KSJyXgWfBhf7xJ1Oi0w5KkLqelaQZGMp/view?usp=sharing)

Place the file in the **same folder** as `Samiksha_HealthcareAnalytics.py` before running.

**Columns:**

| Column | Description |
|---|---|
| Patient Id | Unique patient identifier (dropped — PII) |
| Patient Admission Date | Visit date (day/month/year format) |
| Patient Admission Time | Visit time (12-hour AM/PM format) |
| Merged | Patient name (dropped — PII) |
| Patient Gender | Gender (contains typo "Femaleemale" — fixed) |
| Patient Age | Age in years |
| Patient Race | Race/ethnicity category |
| Department Referral | Referred department ("None" → "No Referral") |
| Patient Admission Flag | Admission / Not Admission |
| Patient Satisfaction Score | 0–10 score (~73% missing, not imputed) |
| Patient Waittime | Wait time in minutes |

---

## 🛠️ Technologies Used

| Category | Library / Tool |
|---|---|
| Data manipulation | pandas 2.2.2, numpy 1.26.4 |
| Visualisation | plotly 5.22.0 |
| Machine Learning | scikit-learn 1.5.0 |
| Trend analysis | statsmodels 0.14.2 |
| Dashboard | Streamlit 1.35.0 |
| Language | Python 3.9+ |

---

## 🤖 Predictive Model

Two models were trained and compared as an **honest experiment**:

- **Logistic Regression** — linear baseline classifier
- **Random Forest** — ensemble tree-based classifier

**Target:** Patient Admission Flag (Admission vs. Not Admission)  
**Features used:** gender, age, race, department, wait time, hour, weekday, age group  
**Important caveat:** Both models perform near the no-skill baseline (~50% accuracy). This is expected because clinical admission decisions depend on diagnostic features (vitals, lab results, diagnoses) that are absent from this dataset. The experiment is included for educational transparency — it shows that feature quality matters more than algorithm choice.

---

## ⚙️ Setup & Run Instructions

### 1. Prerequisites
- Python 3.9 or higher
- pip

### 2. Clone / download the project
Place these files in the same folder:
```
healthcare_data.csv
Samiksha_HealthcareAnalytics.py
requirements.txt
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
streamlit run Samiksha_HealthcareAnalytics.py
```

The app will open automatically at `http://localhost:8501` in your browser.

---

## 📊 Dashboard Pages

| Page | Contents |
|---|---|
| 📊 Executive Overview | KPIs, monthly trend, data quality summary |
| ⏱️ Patient Flow & Wait Time | Weekday/hour charts, heatmap, wait-time analysis |
| 🏬 Departments & Patients | Referrals, admission rates, age/race demographics |
| 😊 Satisfaction Analysis | Score distribution, wait-band analysis, scatter plots |
| 💡 Insights & Model Results | Risks, opportunities, recommended actions, model output |

Sidebar filters: **Date Range**, **Gender**, **Department** — apply globally to all pages.

---

## 🔑 Key Findings

1. **Peak day and hour** — The busiest arrivals cluster on weekdays, with a pronounced morning surge. Staffing should be flexed accordingly.
2. **Wait times vary by department** — Some clinical departments show average waits well above the overall mean; targeted capacity investment is needed there.
3. **59% of visits have no referral** — A majority of patients arrive without a department referral, suggesting walk-in or unclear care-pathway issues.
4. **Only 27% of patients filled in a satisfaction survey** — Any satisfaction averages must be read with this caveat; the 73% non-response is a major data gap.
5. **Higher wait times correlate with lower satisfaction** — The lowest satisfaction scores consistently appear in the longest wait-time bands.
6. **Predictive model is near-chance** — Without clinical features, demographic/logistics data alone cannot reliably predict admission. Both models produced accuracy close to the baseline.

---

## 📁 File Structure

```
healthcare project/
├── healthcare_data.csv                  ← source data (download separately)
├── Samiksha_HealthcareAnalytics.py      ← main app (cleaning + model + dashboard)
├── requirements.txt                     ← Python dependencies
├── README.md                            ← this file
└── Samiksha_ProjectReport.docx          ← full project report
```

---

## ⚠️ Disclaimer

This project is **for educational purposes only**. All insights, charts, and model results are derived from a synthetic/public dataset and do not represent real patient data. They must not be used for any clinical, diagnostic, or hospital management decisions.

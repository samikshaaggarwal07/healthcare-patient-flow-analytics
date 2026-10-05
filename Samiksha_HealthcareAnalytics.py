"""
Samiksha_HealthcareAnalytics.py
================================
Healthcare Patient Flow Analytics — Streamlit Dashboard
Includes: Data Cleaning, EDA, Predictive Modelling, Interactive Dashboard

NOTE: This project is for educational purposes only.
      All data shown is from a synthetic/public dataset.
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ──────────────────────────────────────────────

@st.cache_data
def load_and_clean():
    """Load healthcare_data.csv, clean it, and return both raw and clean DataFrames."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "healthcare_data.csv")
    raw = pd.read_csv(csv_path, encoding="utf-8-sig")

    df = raw.copy()

    # ── Step 1: Rename columns for convenience ──
    df.columns = [
        "patient_id", "admission_date", "admission_time",
        "patient_name", "gender", "age", "race",
        "department", "admission_flag", "satisfaction_score", "wait_time"
    ]

    # ── Step 2: Drop PII columns (patient_id, patient_name) ──
    df.drop(columns=["patient_id", "patient_name"], inplace=True)

    # ── Step 3: Fix gender typo "Femaleemale" → "Female" ──
    df["gender"] = df["gender"].replace("Femaleemale", "Female")

    # ── Step 4: Parse dates (mixed formats — some M/D/YYYY, some D/MM/YYYY) ──
    # Use dayfirst=True to handle DD/MM/YYYY entries; errors='coerce' catches bad values
    df["admission_date"] = pd.to_datetime(df["admission_date"], dayfirst=True, errors="coerce")

    # ── Step 5: Parse time and combine into a single datetime ──
    df["admission_time_parsed"] = pd.to_datetime(df["admission_time"], format="%I:%M:%S %p", errors="coerce")
    df["admission_datetime"] = df["admission_date"] + pd.to_timedelta(
        df["admission_time_parsed"].dt.hour.fillna(0).astype(int), unit="h"
    ) + pd.to_timedelta(
        df["admission_time_parsed"].dt.minute.fillna(0).astype(int), unit="m"
    )
    df.drop(columns=["admission_time", "admission_time_parsed"], inplace=True)

    # ── Step 6: Replace "None" department with "No Referral" ──
    df["department"] = df["department"].replace("None", "No Referral")
    df["department"] = df["department"].fillna("No Referral")

    # ── Step 7: Satisfaction score — keep NaN; do NOT impute ──
    df["satisfaction_score"] = pd.to_numeric(df["satisfaction_score"], errors="coerce")

    # ── Step 8: Ensure numeric columns are correct types ──
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["wait_time"] = pd.to_numeric(df["wait_time"], errors="coerce")

    # ── Step 9: Drop rows where core fields are still missing ──
    before_drop = len(df)
    df = df.dropna(subset=["admission_date", "age", "wait_time", "gender"])
    after_drop = len(df)

    # ── Step 10: Feature Engineering ──
    df["month"] = df["admission_date"].dt.month
    df["month_name"] = df["admission_date"].dt.strftime("%b")
    df["month_year"] = df["admission_date"].dt.to_period("M").astype(str)
    df["weekday"] = df["admission_date"].dt.day_name()
    df["hour"] = df["admission_datetime"].dt.hour

    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 17, 35, 50, 65, 120],
        labels=["Child (0–12)", "Teen (13–17)", "Young Adult (18–35)",
                "Adult (36–50)", "Senior (51–65)", "Elderly (65+)"]
    )

    df["wait_band"] = pd.cut(
        df["wait_time"],
        bins=[0, 10, 20, 30, 40, 50, 60, 200],
        labels=["≤10 min", "11–20 min", "21–30 min",
                "31–40 min", "41–50 min", "51–60 min", "60+ min"]
    )

    df["admitted"] = (df["admission_flag"] == "Admission").astype(int)

    # ── Build a before/after quality summary ──
    quality_summary = {
        "raw_rows": len(raw),
        "clean_rows": len(df),
        "dropped_rows": before_drop - after_drop,
        "gender_typos_fixed": 17,
        "dept_none_relabelled": int((df["department"] == "No Referral").sum()),
        "satisfaction_missing_pct": round(df["satisfaction_score"].isna().mean() * 100, 1),
        "satisfaction_response_rate": round(df["satisfaction_score"].notna().mean() * 100, 1),
        "date_range_start": df["admission_date"].min().strftime("%d %b %Y"),
        "date_range_end": df["admission_date"].max().strftime("%d %b %Y"),
    }

    return df, quality_summary


# ──────────────────────────────────────────────
# 2. MACHINE LEARNING MODEL
# ──────────────────────────────────────────────

@st.cache_data
def run_model(df):
    """
    Predict Patient Admission Flag using Logistic Regression and Random Forest.
    Returns a result dict with metrics and confusion matrices.
    """
    model_df = df[["gender", "age", "race", "department", "wait_time", "hour",
                   "weekday", "age_group", "admitted"]].dropna()

    features = ["gender", "age", "race", "department", "wait_time", "hour", "weekday", "age_group"]
    X = model_df[features].copy()
    y = model_df["admitted"]

    # Encode categoricals
    le_dict = {}
    for col in ["gender", "race", "department", "weekday", "age_group"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    admission_rate = round(y.mean() * 100, 1)

    results = {
        "lr_accuracy": round(accuracy_score(y_test, y_pred_lr) * 100, 1),
        "lr_f1": round(f1_score(y_test, y_pred_lr) * 100, 1),
        "rf_accuracy": round(accuracy_score(y_test, y_pred_rf) * 100, 1),
        "rf_f1": round(f1_score(y_test, y_pred_rf) * 100, 1),
        "lr_cm": confusion_matrix(y_test, y_pred_lr).tolist(),
        "rf_cm": confusion_matrix(y_test, y_pred_rf).tolist(),
        "n_test": len(y_test),
        "admission_rate": admission_rate,
        "class_balance_note": (
            f"Baseline (always-predict-majority): ~{100 - admission_rate:.0f}% accuracy. "
            f"Admission rate in dataset: {admission_rate}%."
        ),
    }
    return results


# ──────────────────────────────────────────────
# 3. HELPER / CHART UTILITIES
# ──────────────────────────────────────────────

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PALETTE = px.colors.qualitative.Set2

def apply_filters(df, date_range, genders, departments):
    mask = (
        (df["admission_date"] >= pd.Timestamp(date_range[0])) &
        (df["admission_date"] <= pd.Timestamp(date_range[1]))
    )
    if genders:
        mask &= df["gender"].isin(genders)
    if departments:
        mask &= df["department"].isin(departments)
    return df[mask]


def kpi_card(col, label, value, delta=None, help_text=None):
    with col:
        st.metric(label=label, value=value, delta=delta, help=help_text)


# ──────────────────────────────────────────────
# 4. STREAMLIT APP
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Healthcare Patient Flow Analytics",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ── Load data ──
    df, quality = load_and_clean()

    # ── Sidebar ──
    st.sidebar.image(
        "https://img.icons8.com/color/96/hospital.png",
        width=60
    )
    st.sidebar.title("🏥 Healthcare Analytics")
    st.sidebar.markdown("---")

    # Date filter
    min_date = df["admission_date"].min().date()
    max_date = df["admission_date"].max().date()
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d_start, d_end = date_range
    else:
        d_start, d_end = min_date, max_date

    # Gender filter
    all_genders = sorted(df["gender"].dropna().unique().tolist())
    sel_genders = st.sidebar.multiselect("👤 Gender", all_genders, default=all_genders)

    # Department filter
    all_depts = sorted(df["department"].unique().tolist())
    sel_depts = st.sidebar.multiselect("🏬 Department", all_depts, default=all_depts)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ **Educational Use Only** — This dashboard is built for educational "
        "purposes using a synthetic/public healthcare dataset. It does not represent "
        "real patient data and should not be used for clinical decisions."
    )

    # Apply filters
    fdf = apply_filters(df, (d_start, d_end), sel_genders, sel_depts)

    if fdf.empty:
        st.warning("No data matches the selected filters. Please adjust your selections.")
        return

    # ── Navigation ──
    pages = [
        "📊 Executive Overview",
        "⏱️ Patient Flow & Wait Time",
        "🏬 Departments & Patients",
        "😊 Satisfaction Analysis",
        "💡 Insights & Model Results"
    ]
    page = st.sidebar.radio("Navigate to", pages)

    # ════════════════════════════════════════════
    # PAGE 1: EXECUTIVE OVERVIEW
    # ════════════════════════════════════════════
    if page == "📊 Executive Overview":
        st.title("📊 Executive Overview")
        st.markdown(f"*Data range shown: {d_start} → {d_end} | {len(fdf):,} visits after filters*")

        # ── KPIs ──
        total_visits = len(fdf)
        admission_rate = fdf["admitted"].mean() * 100
        avg_wait = fdf["wait_time"].mean()
        sat_df = fdf["satisfaction_score"].dropna()
        avg_sat = sat_df.mean() if len(sat_df) > 0 else float("nan")
        response_rate = fdf["satisfaction_score"].notna().mean() * 100
        referral_pct = (fdf["department"] != "No Referral").mean() * 100

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        kpi_card(k1, "🧑‍⚕️ Total Visits", f"{total_visits:,}")
        kpi_card(k2, "🏥 Admission Rate", f"{admission_rate:.1f}%")
        kpi_card(k3, "⏱️ Avg Wait Time", f"{avg_wait:.1f} min")
        kpi_card(k4, "😊 Avg Satisfaction", f"{avg_sat:.2f}/10" if not np.isnan(avg_sat) else "N/A",
                 help_text="Calculated only on rows with a score")
        kpi_card(k5, "📋 Survey Response Rate", f"{response_rate:.1f}%",
                 help_text="Only 27% of patients provided a satisfaction score")
        kpi_card(k6, "🔗 Referral Rate", f"{referral_pct:.1f}%")

        st.markdown("---")

        # ── Monthly Visits Trend ──
        st.subheader("📈 Monthly Visits Trend")
        monthly = fdf.groupby("month_year").size().reset_index(name="visits")
        monthly = monthly.sort_values("month_year")
        fig_trend = px.line(
            monthly, x="month_year", y="visits",
            markers=True, title="Monthly Patient Visits",
            labels={"month_year": "Month", "visits": "Number of Visits"},
            color_discrete_sequence=["#3B82D4"]
        )
        fig_trend.update_layout(xaxis_tickangle=-45, height=380)
        st.plotly_chart(fig_trend, use_container_width=True)

        # ── Data Quality Summary ──
        st.markdown("---")
        st.subheader("🧹 Data Quality Summary (Before → After Cleaning)")
        q_cols = st.columns(3)
        with q_cols[0]:
            st.info(f"**Raw rows loaded:** {quality['raw_rows']:,}\n\n"
                    f"**Clean rows used:** {quality['clean_rows']:,}\n\n"
                    f"**Rows dropped (missing core fields):** {quality['dropped_rows']}")
        with q_cols[1]:
            st.info(f"**Gender typos fixed:** {quality['gender_typos_fixed']} ('Femaleemale' → 'Female')\n\n"
                    f"**'None' dept relabelled 'No Referral':** {quality['dept_none_relabelled']:,}\n\n"
                    f"**Date range:** {quality['date_range_start']} – {quality['date_range_end']}")
        with q_cols[2]:
            st.info(f"**Satisfaction score missing:** {quality['satisfaction_missing_pct']}% of rows\n\n"
                    f"**Survey response rate:** {quality['satisfaction_response_rate']}%\n\n"
                    f"**Satisfaction: NOT imputed** — analysis uses only real responses")

        st.caption("PII columns (Patient Id, Patient Name) were dropped and are not shown anywhere in this dashboard.")

    # ════════════════════════════════════════════
    # PAGE 2: PATIENT FLOW & WAIT TIME
    # ════════════════════════════════════════════
    elif page == "⏱️ Patient Flow & Wait Time":
        st.title("⏱️ Patient Flow & Wait Time Analysis")

        # ── Visits by Weekday ──
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Visits by Weekday")
            wd = fdf.groupby("weekday").size().reindex(WEEKDAY_ORDER).reset_index()
            wd.columns = ["Weekday", "Visits"]
            fig_wd = px.bar(wd, x="Weekday", y="Visits", color="Visits",
                            color_continuous_scale="Blues",
                            title="Patient Visits by Day of Week")
            fig_wd.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_wd, use_container_width=True)

        with col2:
            st.subheader("Visits by Hour of Day")
            hr = fdf.groupby("hour").size().reset_index(name="Visits")
            fig_hr = px.bar(hr, x="hour", y="Visits", color="Visits",
                            color_continuous_scale="Greens",
                            title="Patient Visits by Hour of Day",
                            labels={"hour": "Hour (24h)"})
            fig_hr.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_hr, use_container_width=True)

        # ── Heatmap: Weekday × Hour ──
        st.subheader("🔥 Peak-Load Heatmap (Weekday × Hour)")
        heat = fdf.groupby(["weekday", "hour"]).size().reset_index(name="visits")
        heat_pivot = heat.pivot(index="weekday", columns="hour", values="visits").fillna(0)
        heat_pivot = heat_pivot.reindex([d for d in WEEKDAY_ORDER if d in heat_pivot.index])
        fig_heat = px.imshow(
            heat_pivot,
            labels=dict(x="Hour of Day", y="Weekday", color="Visits"),
            color_continuous_scale="YlOrRd",
            title="Visit Volume: Day vs Hour Heatmap",
            aspect="auto"
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Wait Time by Department ──
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Wait Time by Department")
            wt_dept = fdf.groupby("department")["wait_time"].mean().reset_index()
            wt_dept.columns = ["Department", "Avg Wait (min)"]
            wt_dept = wt_dept.sort_values("Avg Wait (min)", ascending=True)
            fig_wt_dept = px.bar(
                wt_dept, x="Avg Wait (min)", y="Department", orientation="h",
                title="Average Wait Time by Department",
                color="Avg Wait (min)", color_continuous_scale="Reds"
            )
            fig_wt_dept.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_wt_dept, use_container_width=True)

        with col4:
            st.subheader("Wait Time by Age Group")
            wt_age = fdf.groupby("age_group", observed=True)["wait_time"].mean().reset_index()
            wt_age.columns = ["Age Group", "Avg Wait (min)"]
            fig_wt_age = px.bar(
                wt_age, x="Age Group", y="Avg Wait (min)",
                title="Average Wait Time by Age Group",
                color="Avg Wait (min)", color_continuous_scale="Purples"
            )
            fig_wt_age.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_wt_age, use_container_width=True)

        # ── Wait Time Distribution ──
        st.subheader("Wait Time Distribution")
        fig_wt_dist = px.histogram(
            fdf, x="wait_time", nbins=40,
            title="Distribution of Patient Wait Times",
            labels={"wait_time": "Wait Time (minutes)"},
            color_discrete_sequence=["#7C5CD8"]
        )
        fig_wt_dist.update_layout(height=350)
        st.plotly_chart(fig_wt_dist, use_container_width=True)

    # ════════════════════════════════════════════
    # PAGE 3: DEPARTMENTS & PATIENTS
    # ════════════════════════════════════════════
    elif page == "🏬 Departments & Patients":
        st.title("🏬 Departments & Patient Demographics")

        # ── Referrals by Department ──
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Referrals by Department")
            dept_counts = fdf.groupby("department").size().reset_index(name="Patients")
            dept_counts = dept_counts.sort_values("Patients", ascending=False)
            fig_dept = px.bar(
                dept_counts, x="department", y="Patients",
                color="department", title="Patient Referrals by Department",
                labels={"department": "Department"},
                color_discrete_sequence=PALETTE
            )
            fig_dept.update_layout(showlegend=False, height=380, xaxis_tickangle=-20)
            st.plotly_chart(fig_dept, use_container_width=True)

        with col2:
            st.subheader("Admission Rate by Department")
            adm_dept = fdf.groupby("department")["admitted"].mean().reset_index()
            adm_dept.columns = ["Department", "Admission Rate"]
            adm_dept["Admission Rate (%)"] = (adm_dept["Admission Rate"] * 100).round(1)
            adm_dept = adm_dept.sort_values("Admission Rate (%)", ascending=False)
            fig_adm_dept = px.bar(
                adm_dept, x="Department", y="Admission Rate (%)",
                color="Admission Rate (%)", color_continuous_scale="Blues",
                title="Admission Rate by Department (%)"
            )
            fig_adm_dept.update_layout(height=380, coloraxis_showscale=False, xaxis_tickangle=-20)
            st.plotly_chart(fig_adm_dept, use_container_width=True)

        # ── Age Group Distribution ──
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Visits by Age Group")
            age_grp = fdf.groupby("age_group", observed=True).size().reset_index(name="Visits")
            fig_age = px.pie(
                age_grp, names="age_group", values="Visits",
                title="Patient Visits by Age Group",
                color_discrete_sequence=PALETTE,
                hole=0.35
            )
            fig_age.update_layout(height=380)
            st.plotly_chart(fig_age, use_container_width=True)

        with col4:
            st.subheader("Gender Distribution")
            gen = fdf["gender"].value_counts().reset_index()
            gen.columns = ["Gender", "Count"]
            fig_gen = px.pie(
                gen, names="Gender", values="Count",
                title="Patient Gender Split",
                color_discrete_sequence=["#3B82D4", "#7C5CD8", "#F59E0B"],
                hole=0.35
            )
            fig_gen.update_layout(height=380)
            st.plotly_chart(fig_gen, use_container_width=True)

        # ── Race Distribution ──
        st.subheader("Visits by Race / Ethnicity")
        race = fdf["race"].value_counts().reset_index()
        race.columns = ["Race", "Count"]
        fig_race = px.bar(
            race, x="Race", y="Count",
            title="Patient Visits by Race / Ethnicity",
            color="Race", color_discrete_sequence=PALETTE
        )
        fig_race.update_layout(showlegend=False, height=380, xaxis_tickangle=-20)
        st.plotly_chart(fig_race, use_container_width=True)

        # ── Age Distribution Histogram ──
        st.subheader("Age Distribution")
        fig_age_hist = px.histogram(
            fdf, x="age", nbins=30,
            title="Age Distribution of Patients",
            labels={"age": "Age (years)"},
            color_discrete_sequence=["#10B981"]
        )
        fig_age_hist.update_layout(height=350)
        st.plotly_chart(fig_age_hist, use_container_width=True)

    # ════════════════════════════════════════════
    # PAGE 4: SATISFACTION ANALYSIS
    # ════════════════════════════════════════════
    elif page == "😊 Satisfaction Analysis":
        st.title("😊 Patient Satisfaction Analysis")

        sat_df_page = fdf[fdf["satisfaction_score"].notna()].copy()
        total_with_score = len(sat_df_page)
        response_rate_page = len(sat_df_page) / len(fdf) * 100 if len(fdf) > 0 else 0

        if total_with_score == 0:
            st.warning("No satisfaction data available for the selected filters.")
            return

        st.info(
            f"ℹ️ **Survey Response Rate: {response_rate_page:.1f}%** — Only {total_with_score:,} out of "
            f"{len(fdf):,} filtered patients provided a satisfaction score. "
            f"The analysis below is based solely on these responses."
        )

        # ── Score Distribution ──
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Satisfaction Score Distribution")
            score_counts = sat_df_page["satisfaction_score"].value_counts().sort_index().reset_index()
            score_counts.columns = ["Score", "Count"]
            fig_score = px.bar(
                score_counts, x="Score", y="Count",
                title="Distribution of Satisfaction Scores (0–10)",
                color="Score", color_continuous_scale="RdYlGn",
                labels={"Score": "Satisfaction Score"}
            )
            fig_score.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_score, use_container_width=True)

        with col2:
            st.subheader("Avg Satisfaction by Wait-Time Band")
            sat_band = sat_df_page.groupby("wait_band", observed=True)["satisfaction_score"].mean().reset_index()
            sat_band.columns = ["Wait Band", "Avg Score"]
            fig_sat_band = px.bar(
                sat_band, x="Wait Band", y="Avg Score",
                title="Avg Satisfaction Score by Wait-Time Band",
                color="Avg Score", color_continuous_scale="RdYlGn",
                range_color=[0, 10]
            )
            fig_sat_band.update_layout(height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_sat_band, use_container_width=True)

        # ── By Department & Age Group ──
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Avg Satisfaction by Department")
            sat_dept = sat_df_page.groupby("department")["satisfaction_score"].agg(
                ["mean", "count"]).reset_index()
            sat_dept.columns = ["Department", "Avg Score", "Responses"]
            sat_dept = sat_dept.sort_values("Avg Score", ascending=False)
            fig_sat_dept = px.bar(
                sat_dept, x="Department", y="Avg Score",
                title="Avg Satisfaction Score by Department",
                color="Avg Score", color_continuous_scale="RdYlGn",
                range_color=[0, 10],
                hover_data=["Responses"]
            )
            fig_sat_dept.update_layout(height=380, coloraxis_showscale=False, xaxis_tickangle=-20)
            st.plotly_chart(fig_sat_dept, use_container_width=True)

        with col4:
            st.subheader("Avg Satisfaction by Age Group")
            sat_age = sat_df_page.groupby("age_group", observed=True)["satisfaction_score"].mean().reset_index()
            sat_age.columns = ["Age Group", "Avg Score"]
            fig_sat_age = px.bar(
                sat_age, x="Age Group", y="Avg Score",
                title="Avg Satisfaction Score by Age Group",
                color="Avg Score", color_continuous_scale="RdYlGn",
                range_color=[0, 10]
            )
            fig_sat_age.update_layout(height=380, coloraxis_showscale=False)
            st.plotly_chart(fig_sat_age, use_container_width=True)

        # ── Scatter: Wait Time vs Satisfaction ──
        st.subheader("Wait Time vs. Satisfaction Score")
        fig_scatter = px.scatter(
            sat_df_page, x="wait_time", y="satisfaction_score",
            color="department", trendline="ols",
            title="Wait Time vs. Satisfaction Score (with trend line)",
            labels={"wait_time": "Wait Time (min)", "satisfaction_score": "Satisfaction Score"},
            opacity=0.5,
            color_discrete_sequence=PALETTE
        )
        fig_scatter.update_layout(height=420)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Trend line shown per department. A downward slope suggests longer waits reduce satisfaction.")

    # ════════════════════════════════════════════
    # PAGE 5: INSIGHTS & MODEL RESULTS
    # ════════════════════════════════════════════
    elif page == "💡 Insights & Model Results":
        st.title("💡 Insights, Risks & Predictive Model Results")

        # ── Compute real numbers from FULL dataset (not filtered) ──
        busiest_day = df.groupby("weekday").size().idxmax()
        busiest_hour = df.groupby("hour").size().idxmax()
        longest_wait_dept = df.groupby("department")["wait_time"].mean().idxmax()
        longest_wait_val = df.groupby("department")["wait_time"].mean().max()
        avg_wait_overall = df["wait_time"].mean()
        top_dept = df[df["department"] != "No Referral"].groupby("department").size().idxmax()
        sat_by_band = df[df["satisfaction_score"].notna()].groupby("wait_band", observed=True)[
            "satisfaction_score"].mean()
        worst_band = sat_by_band.idxmin() if len(sat_by_band) > 0 else "N/A"
        best_band = sat_by_band.idxmax() if len(sat_by_band) > 0 else "N/A"
        admission_rate_full = df["admitted"].mean() * 100

        st.markdown("---")
        st.subheader("🔴 Risks")
        st.markdown(f"""
- **Peak overload:** The busiest day is **{busiest_day}** and the busiest hour is **{busiest_hour}:00**.
  Staff and resource allocation should anticipate this load.
- **Longest waits:** The **{longest_wait_dept}** department has the highest average wait time
  (**{longest_wait_val:.1f} min** vs. overall average of **{avg_wait_overall:.1f} min**).
  Extended waits risk patient dissatisfaction and adverse outcomes.
- **Low survey response rate (27%):** Satisfaction scores are available for only ~27% of patients.
  This creates a significant non-response bias — the remaining 73% may have systematically
  different experiences that are invisible in the data.
- **'No Referral' dominates (~59% of visits):** A large share of patients arrive without a
  department referral, which may indicate unclear care pathways or walk-in overload.
- **Satisfaction drops with wait time:** Patients in the **{worst_band}** wait-time band report
  the lowest average satisfaction scores.
""")

        st.subheader("🟢 Opportunities")
        st.markdown(f"""
- **Targeted staffing on {busiest_day} at {busiest_hour}:00** can directly reduce peak-hour queues.
- **{top_dept}** is the most referred clinical department — investing in its capacity
  (beds, staff, equipment) would benefit the highest patient volume.
- **Satisfaction in the {best_band} wait-time band is the highest** — protocols that reduce
  wait times to this range should be standardized across departments.
- **Closing the survey gap:** Improving satisfaction survey uptake (e.g., SMS/kiosk reminders)
  would provide statistically reliable data for management decisions.
- **Admission rate is {admission_rate_full:.1f}%** — this leaves room to explore whether
  some admissions could be safely managed through enhanced outpatient pathways.
""")

        st.subheader("🔵 Recommended Actions")
        st.markdown(f"""
1. **Flex staffing model:** Deploy additional triage nurses and support staff on {busiest_day}s
   between 09:00–17:00 to address the peak-hour surge.
2. **Wait-time reduction plan for {longest_wait_dept}:** Conduct a root-cause analysis
   (staffing ratio, equipment bottlenecks, scheduling gaps) and set a 90-day target to reduce
   average wait from {longest_wait_val:.1f} min to below {avg_wait_overall:.1f} min.
3. **Referral pathway review:** Investigate why 59% of patients arrive without a referral.
   Consider fast-track referral programs with GPs and community clinics.
4. **Mandatory satisfaction survey:** Implement automatic post-visit surveys (digital/SMS)
   to raise the response rate from 27% to at least 60%.
5. **Monitor monthly:** Track KPIs (admission rate, average wait, satisfaction) monthly
   using this dashboard to identify regressions early.
""")

        # ── Model Results ──
        st.markdown("---")
        st.subheader("🤖 Predictive Model Experiment: Predicting Patient Admission")
        st.markdown(
            "We ran a controlled experiment to test whether demographic and wait-time features "
            "can predict whether a patient will be admitted. **This is an honest experiment — "
            "results are reported without inflation.**"
        )

        with st.spinner("Training models (runs once, then cached)..."):
            model_results = run_model(df)

        col_lr, col_rf = st.columns(2)

        with col_lr:
            st.markdown("#### 📐 Logistic Regression")
            m1, m2 = st.columns(2)
            m1.metric("Accuracy", f"{model_results['lr_accuracy']}%")
            m2.metric("F1 Score", f"{model_results['lr_f1']}%")
            cm_lr = np.array(model_results["lr_cm"])
            fig_cm_lr = px.imshow(
                cm_lr,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not Admitted", "Admitted"],
                y=["Not Admitted", "Admitted"],
                color_continuous_scale="Blues",
                title="Logistic Regression — Confusion Matrix",
                text_auto=True
            )
            fig_cm_lr.update_layout(height=320)
            st.plotly_chart(fig_cm_lr, use_container_width=True)

        with col_rf:
            st.markdown("#### 🌲 Random Forest")
            m3, m4 = st.columns(2)
            m3.metric("Accuracy", f"{model_results['rf_accuracy']}%")
            m4.metric("F1 Score", f"{model_results['rf_f1']}%")
            cm_rf = np.array(model_results["rf_cm"])
            fig_cm_rf = px.imshow(
                cm_rf,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Not Admitted", "Admitted"],
                y=["Not Admitted", "Admitted"],
                color_continuous_scale="Greens",
                title="Random Forest — Confusion Matrix",
                text_auto=True
            )
            fig_cm_rf.update_layout(height=320)
            st.plotly_chart(fig_cm_rf, use_container_width=True)

        st.markdown("---")
        st.subheader("📢 Honest Model Interpretation")
        st.warning(f"""
**⚠️ Important: The model performance is close to the no-skill baseline.**

{model_results['class_balance_note']}

- **Logistic Regression:** {model_results['lr_accuracy']}% accuracy, F1 = {model_results['lr_f1']}%
- **Random Forest:** {model_results['rf_accuracy']}% accuracy, F1 = {model_results['rf_f1']}%
- **Test set size:** {model_results['n_test']:,} patients

**Why is this happening?**
The available features — age, gender, race, department, wait time, day, and hour —
carry **very little predictive signal** for admission. In a real hospital, admission decisions
are driven by clinical factors (vital signs, diagnosis codes, lab results, physician assessment)
that are **not present in this dataset**. The model is essentially learning a weak correlation
from demographic and logistics data, which is insufficient for reliable clinical prediction.

**Conclusion:** Do not use this model for clinical decisions. Its value here is
pedagogical — it demonstrates that high accuracy requires the right features,
not just more data or a better algorithm.
""")

        st.markdown("---")
        st.caption(
            "⚠️ **Educational Purposes Only** — This entire project, including the dashboard, "
            "charts, and model, was built for learning and portfolio demonstration. "
            "It uses a synthetic/public dataset and is not intended for real clinical use."
        )


if __name__ == "__main__":
    main()

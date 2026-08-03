"""
TalentPulse — Interactive HR Attrition Intelligence Dashboard
================================================================
Run: streamlit run streamlit_app.py
"""
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "talentpulse.db"
MODEL_PATH = BASE_DIR / "models" / "xgb_attrition_model.pkl"

st.set_page_config(page_title="TalentPulse | HR Attrition Intelligence", page_icon="💠", layout="wide")

# ---------------------------------------------------------------
# Styling — smart, animated, burnt-orange (#C05800) brand theme
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Poppins:wght@600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(120deg, #FFF8ED 0%, #FDEBD8 30%, #FCE0C4 60%, #FDEBD8 100%);
    background-size: 300% 300%;
    animation: bgShift 18s ease infinite;
}

@keyframes bgShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 4px 18px rgba(192,88,0,0.10); }
    50% { box-shadow: 0 4px 28px rgba(192,88,0,0.32); }
}
@keyframes shimmer {
    0% { background-position: -300px 0; }
    100% { background-position: 300px 0; }
}
@keyframes underlineGrow {
    from { width: 0; }
    to { width: 46px; }
}
@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-4px); }
}

.hero-title {
    font-family: 'Poppins', sans-serif;
    font-size: 2.5rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #7A3800, #C05800 45%, #F08A24 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: fadeInUp 0.7s ease-out;
    display: inline-block;
}
.hero-title .hero-icon { display: inline-block; animation: floatIcon 3s ease-in-out infinite; }
.hero-sub {
    color: #7A4A20; font-size: 0.95rem; margin-top: -6px;
    animation: fadeInUp 0.9s ease-out;
}
.hero-author {
    color: #C05800; font-size: 0.85rem; font-weight: 700; margin-top: 4px;
    animation: fadeInUp 1.05s ease-out;
}

.metric-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(253,235,216,0.92));
    border: 1px solid rgba(192,88,0,0.28);
    border-radius: 16px; padding: 18px 20px;
    animation: fadeInUp 0.6s ease-out, pulseGlow 4s ease-in-out infinite;
    backdrop-filter: blur(6px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.metric-card:hover { transform: translateY(-5px) scale(1.015); }
.metric-label { color: #C05800; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 1px; font-weight: 700; }
.metric-value { font-size: 1.9rem; font-weight: 800; color: #5A2A00; }
.metric-delta-up { color: #dc2626; font-size: 0.85rem; }
.metric-delta-down { color: #16a34a; font-size: 0.85rem; }

.risk-critical { color: #dc2626; font-weight: 700; }
.risk-high { color: #C05800; font-weight: 700; }
.risk-medium { color: #ca8a04; font-weight: 700; }
.risk-low { color: #16a34a; font-weight: 700; }

.section-header {
    position: relative;
    font-family: 'Poppins', sans-serif;
    font-size: 1.15rem; font-weight: 700; color: #5A2A00; margin-top: 14px; margin-bottom: 6px;
    border-left: 4px solid #C05800; padding-left: 10px; animation: fadeInUp 0.5s ease-out;
}
.section-header::after {
    content: ""; display: block; height: 3px; margin-top: 4px; margin-left: 10px;
    background: linear-gradient(90deg, #C05800, #F08A24);
    border-radius: 2px; animation: underlineGrow 0.8s ease-out forwards;
}

.badge {
    display:inline-block; padding: 5px 14px; border-radius: 999px; font-size: 0.75rem;
    font-weight: 700; color:#ffffff; border: 1px solid rgba(192,88,0,0.4);
    background: linear-gradient(90deg, #C05800 0%, #F08A24 50%, #C05800 100%);
    background-size: 300px 100%;
    animation: shimmer 3.5s linear infinite;
}

.footer-credit {
    text-align:center; color:#C05800; font-size:0.85rem; font-weight:600; margin-top: 14px;
}

div[data-testid="stMetricValue"] { animation: fadeInUp 0.6s ease-out; }

/* Streamlit tab styling to match the brand */
button[data-baseweb="tab"] { font-weight: 600; color: #7A4A20; }
button[data-baseweb="tab"][aria-selected="true"] { color: #C05800 !important; }
div[data-baseweb="tab-highlight"] { background-color: #C05800 !important; }

/* Sliders + inputs accent */
div[data-testid="stSlider"] div[role="slider"] { background-color: #C05800 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_resource
def get_writable_db_path():
    """
    Streamlit Community Cloud mounts the repo read-only, but SQLite needs
    to write a small lock/journal file next to the .db even for plain
    reads. Copy the shipped DB into a writable temp dir.

    Always overwrite (not "copy only if missing") — if Streamlit Cloud
    reuses an existing container across a redeploy instead of a fully
    fresh one, a stale copy could otherwise sit in /tmp and silently
    serve outdated predictions after a data update.
    """
    import shutil
    import tempfile
    writable_path = Path(tempfile.gettempdir()) / "talentpulse.db"
    shutil.copy(DB_PATH, writable_path)
    return str(writable_path)


@st.cache_data(ttl=600)
def load_data():
    conn = sqlite3.connect(get_writable_db_path())
    employees = pd.read_sql("SELECT * FROM employees", conn)
    departments = pd.read_sql("SELECT * FROM departments", conn)
    job_roles = pd.read_sql("SELECT * FROM job_roles", conn)
    preds = pd.read_sql("SELECT * FROM attrition_predictions", conn)
    attrition = pd.read_sql("SELECT * FROM attrition_events", conn)
    sentiment = pd.read_sql("SELECT * FROM sentiment_scores", conn)
    engagement = pd.read_sql("SELECT * FROM engagement_surveys", conn)
    conn.close()

    emp = employees.merge(departments, on="department_id").merge(
        job_roles.drop(columns=["department_id"]), on="job_role_id"
    )
    emp = emp.merge(preds, on="employee_id", how="left")

    emp["hire_date"] = pd.to_datetime(emp["hire_date"])
    emp["tenure_months"] = ((pd.Timestamp.today() - emp["hire_date"]).dt.days / 30.44).round(1)

    return emp, attrition, sentiment, engagement


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["feature_cols"]


emp, attrition, sentiment, engagement = load_data()
model, feature_cols = load_model()

RISK_COLORS = {"Low": "#16a34a", "Medium": "#F0A83C", "High": "#C05800", "Critical": "#8B1E00"}
ORANGE_PALETTE = ["#C05800", "#F08A24", "#7A3800", "#F5B85C", "#8B1E00", "#D97B29"]

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.markdown('<div class="hero-title"><span class="hero-icon">💠</span> TalentPulse</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Predictive HR Attrition Intelligence — live from MySQL/SQLite · XGBoost + SHAP · Cox Survival · NLP Sentiment</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-author">Built by Pawan Dubey · Full-Stack &amp; AIML Developer</div>', unsafe_allow_html=True)
st.write("")

tabs = st.tabs(["📊 Executive Overview", "🎯 Flight-Risk Watchlist", "⏳ Survival Analysis",
                 "💬 Sentiment & Exit Themes", "🔮 What-If Simulator"])

# =================================================================
# TAB 1 — Executive Overview
# =================================================================
with tabs[0]:
    active = emp[emp["is_active"] == 1]
    total_headcount = len(active)
    attrition_rate = len(attrition) / len(emp)
    critical_count = (active["risk_band"] == "Critical").sum()
    avg_risk = active["risk_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in zip(
        [c1, c2, c3, c4],
        ["Active Headcount", "Historical Attrition Rate", "Critical Risk Employees", "Avg. Predicted Risk"],
        [f"{total_headcount:,}", f"{attrition_rate:.1%}", f"{critical_count:,}", f"{avg_risk:.1%}"],
        ["current workforce", "last 24 months", "needs action now", "across active staff"],
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div style="color:#64748b; font-size:0.78rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-header">Attrition Rate by Department</div>', unsafe_allow_html=True)

    dept_attr = emp.groupby("department_name").agg(
        headcount=("employee_id", "count"),
        exits=("is_active", lambda x: (x == 0).sum())
    ).reset_index()
    dept_attr["attrition_rate"] = dept_attr["exits"] / dept_attr["headcount"]

    fig = px.bar(
        dept_attr.sort_values("attrition_rate", ascending=True),
        x="attrition_rate", y="department_name", orientation="h",
        color="attrition_rate", color_continuous_scale=["#16a34a", "#F0A83C", "#C05800", "#8B1E00"],
        text=dept_attr.sort_values("attrition_rate", ascending=True)["attrition_rate"].apply(lambda x: f"{x:.1%}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        height=350, showlegend=False, coloraxis_showscale=False,
        xaxis_title="Attrition Rate", yaxis_title="",
        transition={"duration": 700, "easing": "cubic-in-out"},
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">Risk Band Distribution</div>', unsafe_allow_html=True)
        band_counts = active["risk_band"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0)
        fig2 = go.Figure(go.Pie(
            labels=band_counts.index, values=band_counts.values, hole=0.55,
            marker=dict(colors=[RISK_COLORS[b] for b in band_counts.index]),
            pull=[0.05 if b == "Critical" else 0 for b in band_counts.index],
            textinfo="label+percent",
        ))
        fig2.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            height=340, showlegend=False,
                            annotations=[dict(text="Risk", x=0.5, y=0.5, font_size=18, showarrow=False, font_color="#5A2A00")])
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Flight-Risk Personas</div>', unsafe_allow_html=True)
        cluster_counts = active["cluster_label"].value_counts().reset_index()
        cluster_counts.columns = ["persona", "count"]
        fig3 = px.bar(cluster_counts, x="count", y="persona", orientation="h",
                       color="persona", color_discrete_sequence=ORANGE_PALETTE)
        fig3.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            height=340, showlegend=False, yaxis_title="", xaxis_title="Employees")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Salary Growth vs Satisfaction (bubble = overtime hours)</div>', unsafe_allow_html=True)
    st.caption("Animated by risk band — toggle legend items to isolate a group.")
    scatter_df = active.copy()
    fig4 = px.scatter(
        scatter_df, x="tenure_months", y="risk_score", size="avg_overtime" if "avg_overtime" in scatter_df else None,
        color="risk_band", color_discrete_map=RISK_COLORS,
        hover_data=["first_name", "last_name", "department_name", "job_role_name"],
        labels={"tenure_months": "Tenure (months)", "risk_score": "Predicted Attrition Risk"},
    ) if "avg_overtime" in scatter_df.columns else None
    if fig4 is None:
        # fallback without size dim if not present
        fig4 = px.scatter(scatter_df, x="tenure_months", y="risk_score", color="risk_band",
                           color_discrete_map=RISK_COLORS,
                           hover_data=["first_name", "last_name", "department_name"])
    fig4.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=420)
    st.plotly_chart(fig4, use_container_width=True)


# =================================================================
# TAB 2 — Flight-Risk Watchlist
# =================================================================
with tabs[1]:
    st.markdown('<div class="section-header">Live Flight-Risk Watchlist</div>', unsafe_allow_html=True)
    min_score = st.slider("Minimum risk score", 0.0, 1.0, 0.10, 0.01)
    dept_filter = st.multiselect("Filter by department", sorted(emp["department_name"].unique()))

    watch = emp[(emp["is_active"] == 1) & (emp["risk_score"] >= min_score)]
    if dept_filter:
        watch = watch[watch["department_name"].isin(dept_filter)]
    watch = watch.sort_values("risk_score", ascending=False)

    fallback_used = False
    if len(watch) == 0:
        fallback_used = True
        watch = emp[emp["is_active"] == 1]
        if dept_filter:
            watch = watch[watch["department_name"].isin(dept_filter)]
        watch = watch.sort_values("risk_score", ascending=False).head(20)

    if fallback_used:
        st.info(f"No one crosses a {min_score:.0%} risk threshold right now — showing the "
                 "20 highest-risk active employees instead. Try lowering the slider to set "
                 "your own cutoff.")
    else:
        st.write(f"**{len(watch)}** employees match this filter.")

    display_cols = ["employee_id", "first_name", "last_name", "department_name", "job_role_name",
                     "risk_score", "risk_band", "top_driver_1", "top_driver_2", "top_driver_3",
                     "predicted_tenure_months", "cluster_label"]
    st.dataframe(
        watch[display_cols].style.format({"risk_score": "{:.1%}"}),
        use_container_width=True, height=420,
    )

    csv = watch[display_cols].to_csv(index=False).encode()
    st.download_button("⬇️ Export watchlist as CSV", csv, "flight_risk_watchlist.csv", "text/csv")


# =================================================================
# TAB 3 — Survival Analysis
# =================================================================
with tabs[2]:
    st.markdown('<div class="section-header">Predicted Tenure Distribution by Department</div>', unsafe_allow_html=True)
    active = emp[emp["is_active"] == 1]
    fig5 = px.box(active, x="department_name", y="predicted_tenure_months", color="department_name",
                   color_discrete_sequence=ORANGE_PALETTE)
    fig5.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=420, showlegend=False, xaxis_title="", yaxis_title="Predicted Remaining Tenure (months)")
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-header">Cumulative Attrition Over Time</div>', unsafe_allow_html=True)
    attr_ts = attrition.copy()
    attr_ts["exit_date"] = pd.to_datetime(attr_ts["exit_date"])
    attr_ts = attr_ts.sort_values("exit_date")
    attr_ts["cumulative_exits"] = range(1, len(attr_ts) + 1)
    fig6 = px.area(attr_ts, x="exit_date", y="cumulative_exits")
    fig6.update_traces(line_color="#C05800", fillcolor="rgba(192,88,0,0.18)")
    fig6.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=380, xaxis_title="", yaxis_title="Cumulative Exits")
    st.plotly_chart(fig6, use_container_width=True)


# =================================================================
# TAB 4 — Sentiment & Exit Themes
# =================================================================
with tabs[3]:
    st.markdown('<div class="section-header">Sentiment Breakdown — Engagement Surveys</div>', unsafe_allow_html=True)
    surv_sent = sentiment[sentiment["source_table"] == "engagement_surveys"]
    sent_counts = surv_sent["sentiment_label"].value_counts()
    fig7 = go.Figure(go.Bar(
        x=sent_counts.index, y=sent_counts.values,
        marker_color=["#4ade80" if l == "Positive" else "#f87171" if l == "Negative" else "#94a3b8" for l in sent_counts.index],
    ))
    fig7.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=340, xaxis_title="", yaxis_title="Survey Responses")
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown('<div class="section-header">Exit Interview Reasons</div>', unsafe_allow_html=True)
    reasons = attrition["exit_reason"].value_counts().reset_index()
    reasons.columns = ["reason", "count"]
    fig8 = px.bar(reasons, x="count", y="reason", orientation="h", color="count",
                   color_continuous_scale=["#F5B85C", "#C05800", "#7A3800"])
    fig8.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        height=380, coloraxis_showscale=False, yaxis_title="", xaxis_title="Exits")
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="section-header">Sample Negative Exit-Interview Comments</div>', unsafe_allow_html=True)
    neg = attrition.merge(
        sentiment[sentiment["source_table"] == "attrition_events"].drop(columns=["employee_id"]),
        left_on="event_id", right_on="source_id"
    )
    neg = neg[neg["sentiment_label"] == "Negative"].head(8)
    for _, r in neg.iterrows():
        st.markdown(f"> *\"{r['exit_interview_text']}\"* — Employee #{r['employee_id']} · {r['exit_reason']}")


# =================================================================
# TAB 5 — What-If Simulator
# =================================================================
with tabs[4]:
    st.markdown('<div class="section-header">🔮 Live What-If Risk Simulator</div>', unsafe_allow_html=True)
    st.caption("Adjust an employee's profile and get a live, explainable attrition risk score from the trained XGBoost model.")

    colL, colR = st.columns([1, 1.2])
    with colL:
        age = st.slider("Age", 21, 60, 32)
        edu = st.slider("Education level (1-5)", 1, 5, 3)
        salary = st.number_input("Monthly salary", 15000, 200000, 45000, step=1000)
        hike = st.slider("Last hike %", 0.0, 15.0, 4.0)
        sal_growth = st.slider("Salary growth over tenure (%)", -10.0, 60.0, 10.0)
        avg_rating = st.slider("Avg performance rating", 1.0, 5.0, 3.2)
        last_rating = st.slider("Last performance rating", 1.0, 5.0, 3.2)
        n_promo = st.slider("Number of promotions", 0, 5, 0)
        avg_sat = st.slider("Avg satisfaction score", 1.0, 5.0, 3.0)
        last_sat = st.slider("Last satisfaction score", 1.0, 5.0, 3.0)
        avg_wlb = st.slider("Avg work-life balance score", 1.0, 5.0, 3.0)
        avg_absent = st.slider("Avg absent days/month", 0.0, 10.0, 1.2)
        avg_ot = st.slider("Avg overtime hours/month", 0.0, 40.0, 8.0)
        tenure = st.slider("Tenure (months)", 1, 240, 36)

    input_row = pd.DataFrame([{
        "age": age, "education_level": edu, "monthly_salary": salary, "hike_percent": hike,
        "salary_growth_pct": sal_growth, "avg_rating": avg_rating, "last_rating": last_rating,
        "n_promotions": n_promo, "avg_satisfaction": avg_sat, "last_satisfaction": last_sat,
        "avg_worklife": avg_wlb, "avg_absent_days": avg_absent, "avg_overtime": avg_ot,
        "tenure_months": tenure,
    }])[feature_cols]

    risk = model.predict_proba(input_row)[0, 1]
    band = "Critical" if risk >= 0.75 else "High" if risk >= 0.5 else "Medium" if risk >= 0.25 else "Low"

    with colR:
        fig9 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk * 100,
            number={"suffix": "%", "font": {"size": 44, "color": "#5A2A00"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                "bar": {"color": RISK_COLORS[band]},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 25], "color": "rgba(22,163,74,0.20)"},
                    {"range": [25, 50], "color": "rgba(240,168,60,0.22)"},
                    {"range": [50, 75], "color": "rgba(192,88,0,0.22)"},
                    {"range": [75, 100], "color": "rgba(139,30,0,0.22)"},
                ],
            },
            title={"text": f"Predicted Attrition Risk — {band}", "font": {"size": 16, "color": "#5A2A00"}},
        ))
        fig9.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", height=340,
                            transition={"duration": 500, "easing": "elastic-in-out"})
        st.plotly_chart(fig9, use_container_width=True)

        st.markdown(f'<span class="badge">Live model inference · XGBoost</span>', unsafe_allow_html=True)

        # local SHAP explanation for this single what-if input
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(input_row)[0]
        contrib = pd.DataFrame({"feature": feature_cols, "impact": sv}).sort_values("impact", key=abs, ascending=False).head(6)
        fig10 = px.bar(contrib, x="impact", y="feature", orientation="h",
                        color="impact", color_continuous_scale=["#16a34a", "#F0A83C", "#C05800"])
        fig10.update_layout(template="plotly_white", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             height=320, coloraxis_showscale=False, yaxis_title="", xaxis_title="SHAP impact on risk",
                             title="Why this score — top drivers")
        st.plotly_chart(fig10, use_container_width=True)

st.write("")
st.markdown('<div class="footer-credit">💠 TalentPulse — Designed &amp; Developed by <b>Pawan Dubey</b> · Python (XGBoost, SHAP, lifelines, VADER) · MySQL/SQLite · Streamlit + Plotly · Power BI companion dashboard</div>', unsafe_allow_html=True)
"""
CLV Dashboard — BG/NBD + Gamma-Gamma

Run locally with:
    streamlit run app.py

Upload a transactions CSV (customer_id, order_date, order_value) and get:
  - fitted model parameters
  - diagnostic plots (calibration/holdout, frequency-recency matrix, etc.)
  - a full CLV table you can filter, sort, and download
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import (
    plot_calibration_purchases_vs_holdout_purchases,
    plot_frequency_recency_matrix,
    plot_period_transactions,
    plot_probability_alive_matrix,
)
from lifetimes.utils import (
    calibration_and_holdout_data,
    summary_data_from_transaction_data,
)

st.set_page_config(page_title="Jason Peng's CLV Dashboard — BG/NBD + Gamma-Gamma", layout="wide")

# ----------------------------------------------------------------------
# Simple password gate
#
# For local use, the password defaults to "changeme" below — edit it directly.
# For deployment (Streamlit Community Cloud, etc.), instead set a secret named
# APP_PASSWORD in Settings -> Secrets, e.g.:
#     APP_PASSWORD = "your-password-here"
# and delete/ignore the DEFAULT_PASSWORD fallback below for better security.
# ----------------------------------------------------------------------

DEFAULT_PASSWORD = "changeme"  # <-- change this before sharing the app


def check_password() -> bool:
    """Returns True if the user has entered the correct password this session."""

    def password_entered():
        correct = st.secrets.get("APP_PASSWORD", DEFAULT_PASSWORD)
        if st.session_state.get("password_input") == correct:
            st.session_state["password_correct"] = True
            del st.session_state["password_input"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("Jason Peng's CLV Dashboard")
    st.caption("BG/NBD + Gamma-Gamma customer lifetime value modeling")
    st.text_input("Password", type="password", key="password_input", on_change=password_entered)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_transactions(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"customer_id", "order_date", "order_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_value"] = pd.to_numeric(df["order_value"], errors="coerce")
    df = df.dropna(subset=["customer_id", "order_date", "order_value"])
    return df


def fig_from_plotter(plot_fn, *args, **kwargs):
    """Run one of lifetimes' plotting helpers and return the current figure."""
    plt.figure(figsize=(7, 5))
    plot_fn(*args, **kwargs)
    fig = plt.gcf()
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def make_synthetic_data(n_customers: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    obs_start = pd.Timestamp("2023-01-01")
    obs_days = 730

    r, alpha = 0.8, 60.0
    a_beta, b_beta = 1.2, 3.5
    mean_spend_shape, mean_spend_scale = 6.0, 12.0
    order_noise_cv = 0.35

    records = []
    for cust_id in range(1, n_customers + 1):
        lam = rng.gamma(r, 1 / alpha)
        p_churn = rng.beta(a_beta, b_beta)
        cust_mean_value = rng.gamma(mean_spend_shape, mean_spend_scale)
        t = rng.integers(0, obs_days - 1)
        alive = True
        while alive and t < obs_days:
            wait = rng.exponential(1 / lam) if lam > 0 else np.inf
            t += wait
            if t >= obs_days:
                break
            order_date = obs_start + pd.Timedelta(days=t)
            order_value = max(
                1.0,
                rng.gamma(1 / order_noise_cv**2, cust_mean_value * order_noise_cv**2),
            )
            records.append((cust_id, order_date.date(), round(order_value, 2)))
            if rng.random() < p_churn:
                alive = False

    return pd.DataFrame(records, columns=["customer_id", "order_date", "order_value"])


# ----------------------------------------------------------------------
# Sidebar — data input & settings
# ----------------------------------------------------------------------

st.sidebar.title("Jason Peng's CLV Dashboard")
st.sidebar.caption("BG/NBD + Gamma-Gamma")

data_source = st.sidebar.radio(
    "Data source", ["Upload CSV", "Use synthetic sample data"], index=0
)

if data_source == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Transactions CSV", type=["csv"],
        help="Must contain columns: customer_id, order_date, order_value",
    )
    raw_df = None
    if uploaded is not None:
        try:
            raw_df = load_transactions(uploaded)
        except Exception as e:
            st.sidebar.error(str(e))
else:
    n_cust = st.sidebar.slider("Synthetic customers", 500, 10000, 3000, step=500)
    raw_df = make_synthetic_data(n_cust)

st.sidebar.divider()
time_unit = st.sidebar.selectbox("Time unit", ["D", "W"], index=0,
                                  help="D = days, W = weeks. Use weeks for very high-frequency purchasing.")
clv_horizon = st.sidebar.number_input("CLV horizon (months)", min_value=1, max_value=60, value=12)
discount_rate = st.sidebar.number_input("Monthly discount rate", min_value=0.0, max_value=0.2,
                                         value=0.01, step=0.005, format="%.3f")
penalizer = st.sidebar.number_input("Penalizer coefficient (regularization)", min_value=0.0,
                                     max_value=1.0, value=0.001, step=0.001, format="%.3f")

run_btn = st.sidebar.button("Run models", type="primary", use_container_width=True,
                             disabled=raw_df is None)

# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------

st.title("Jason Peng's CLV Dashboard")
st.caption("BG/NBD (purchase frequency & churn) + Gamma-Gamma (spend) → CLV")

if raw_df is None:
    st.info("Upload a transactions CSV in the sidebar, or switch to synthetic sample data, "
            "to get started. Required columns: **customer_id**, **order_date**, **order_value**.")
    st.stop()

with st.expander("Preview uploaded data", expanded=False):
    st.write(f"{len(raw_df):,} transactions, {raw_df['customer_id'].nunique():,} unique customers, "
             f"date range {raw_df['order_date'].min().date()} → {raw_df['order_date'].max().date()}")
    st.dataframe(raw_df.head(20), use_container_width=True)

if not run_btn:
    st.info("Adjust settings in the sidebar, then click **Run models**.")
    st.stop()

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

with st.spinner("Building RFM summary and fitting models..."):
    observation_end = raw_df["order_date"].max()

    summary = summary_data_from_transaction_data(
        raw_df, "customer_id", "order_date", monetary_value_col="order_value",
        observation_period_end=observation_end, freq=time_unit,
    )

    bgf = BetaGeoFitter(penalizer_coef=penalizer)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    horizon_periods = clv_horizon * (30 if time_unit == "D" else 4.345)
    summary["predicted_purchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        horizon_periods, summary["frequency"], summary["recency"], summary["T"]
    )
    summary["prob_alive"] = bgf.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"]
    )

    repeat = summary[summary["frequency"] > 0]
    fm_corr = repeat[["frequency", "monetary_value"]].corr().iloc[0, 1] if len(repeat) > 1 else np.nan

    ggf = GammaGammaFitter(penalizer_coef=penalizer)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])

    summary["predicted_avg_value"] = np.nan
    summary.loc[repeat.index, "predicted_avg_value"] = ggf.conditional_expected_average_profit(
        repeat["frequency"], repeat["monetary_value"]
    )
    pop_avg = ggf.conditional_expected_average_profit(
        summary["frequency"], summary["monetary_value"]
    ).mean()
    summary["predicted_avg_value"] = summary["predicted_avg_value"].fillna(pop_avg)

    summary["clv"] = ggf.customer_lifetime_value(
        bgf, summary["frequency"], summary["recency"], summary["T"], summary["monetary_value"],
        time=clv_horizon, freq=time_unit, discount_rate=discount_rate,
    )
    summary = summary.sort_values("clv", ascending=False)

st.success("Models fit successfully.")

# ----------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(summary):,}")
k2.metric("Repeat customers", f"{len(repeat):,}",
          f"{len(repeat)/len(summary)*100:.0f}% of total")
k3.metric(f"Avg predicted {clv_horizon}mo CLV", f"${summary['clv'].mean():,.2f}")
k4.metric("Total predicted CLV (all customers)", f"${summary['clv'].sum():,.0f}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

tab_overview, tab_diag, tab_table = st.tabs(["Model overview", "Diagnostics", "CLV table"])

with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("BG/NBD parameters")
        st.dataframe(bgf.summary, use_container_width=True)
        st.caption("r, alpha describe the population's purchase-rate (Gamma) distribution. "
                   "a, b describe the population's churn-probability (Beta) distribution.")
    with col2:
        st.subheader("Gamma-Gamma parameters")
        st.dataframe(ggf.summary, use_container_width=True)
        corr_ok = abs(fm_corr) < 0.3 if not np.isnan(fm_corr) else True
        st.metric("Frequency ↔ monetary value correlation", f"{fm_corr:.3f}",
                  delta="OK (near 0)" if corr_ok else "Check assumption",
                  delta_color="normal" if corr_ok else "inverse")
        st.caption("Gamma-Gamma assumes spend is independent of purchase frequency. "
                   "Values well above ~0.3 in magnitude mean this assumption may not hold well.")

    st.subheader(f"Top 15 customers by predicted {clv_horizon}-month CLV")
    st.dataframe(
        summary.head(15)[["frequency", "recency", "T", "monetary_value",
                           "predicted_purchases", "prob_alive", "clv"]]
        .style.format({"monetary_value": "${:,.2f}", "clv": "${:,.2f}",
                        "predicted_purchases": "{:.2f}", "prob_alive": "{:.1%}"}),
        use_container_width=True,
    )

with tab_diag:
    st.caption(
        "Calibration/holdout validates predictions against real held-out purchases. "
        "The other plots show how the fitted model behaves across the frequency/recency space."
    )
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Calibration vs. holdout**")
        try:
            calibration_end = observation_end - pd.Timedelta(days=int(horizon_periods) if time_unit == "D" else int(horizon_periods * 7))
            cal_holdout = calibration_and_holdout_data(
                raw_df, "customer_id", "order_date",
                calibration_period_end=calibration_end,
                observation_period_end=observation_end, freq=time_unit,
            )
            bgf_cal = BetaGeoFitter(penalizer_coef=penalizer)
            bgf_cal.fit(cal_holdout["frequency_cal"], cal_holdout["recency_cal"], cal_holdout["T_cal"])
            fig = fig_from_plotter(plot_calibration_purchases_vs_holdout_purchases, bgf_cal, cal_holdout)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Couldn't build calibration/holdout split (often means the date "
                       f"range is too short for the chosen CLV horizon): {e}")

    with c2:
        st.markdown("**Frequency / recency matrix**")
        fig = fig_from_plotter(plot_frequency_recency_matrix, bgf)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Probability alive matrix**")
        fig = fig_from_plotter(plot_probability_alive_matrix, bgf)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c4:
        st.markdown("**Model fit vs. actual (period transactions)**")
        fig = fig_from_plotter(plot_period_transactions, bgf)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab_table:
    st.subheader("Full CLV table")

    min_clv = st.slider("Minimum CLV filter", 0.0, float(summary["clv"].max()), 0.0)
    filtered = summary[summary["clv"] >= min_clv]
    st.caption(f"Showing {len(filtered):,} of {len(summary):,} customers")
    st.dataframe(filtered, use_container_width=True, height=450)

    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf)
    st.download_button(
        "Download CLV table as CSV",
        data=csv_buf.getvalue(),
        file_name="clv_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption("Built by Jason Peng · BG/NBD + Gamma-Gamma CLV modeling")

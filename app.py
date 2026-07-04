"""
CLV Dashboard — BG/NBD + Gamma-Gamma (English / Chinese)

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
# Translations
# ----------------------------------------------------------------------

README_EN = """
### Customer Lifetime Value modeling with BG/NBD + Gamma-Gamma

A web dashboard that predicts how much each of your customers is worth in
the future, using two well-established statistical models from marketing
science. Upload transaction history, get back a ranked list of customers
with their predicted future value.

## What this actually does

**BG/NBD (Beta Geometric / Negative Binomial Distribution)**
Looks at *when* and *how often* each customer has purchased, and predicts:
- How many purchases they'll make in a future period
- The probability they're still an active customer (vs. having silently
  churned — there's no "cancel" button in retail, so this has to be
  inferred)

**Gamma-Gamma**
Looks at *how much* customers spend per order, and predicts their expected
average future order value. Deliberately kept separate from BG/NBD because
purchase frequency and order size are assumed to be independent.

**Combined:** expected future purchases × expected order value × a
discount rate = predicted Customer Lifetime Value (CLV) for every
customer.

These models are built for **non-contractual, repeat-purchase businesses**
(ecommerce, retail, restaurants) — not subscriptions, where churn is
explicit and better modeled differently.

## What data you need

Just transaction-level records, one row per order:

| customer_id | order_date | order_value |
|---|---|---|
| 1001 | 2024-01-15 | 45.00 |
| 1001 | 2024-03-02 | 62.50 |
| 1002 | 2024-01-20 | 30.00 |

That's genuinely it — three columns. From this we derive the four summary
stats the models actually consume, per customer:

- **frequency** — number of *repeat* purchases (transactions − 1)
- **recency** — time between first and last purchase
- **T** — time between first purchase and "now" (end of observation window)
- **monetary_value** — average value of repeat transactions

### A few things worth deciding upfront

- **Time unit**: days? weeks? Affects interpretability of results.
- **How far back does your data go**, and do you want to hold out a recent
  period to validate predictions against?
- **Do you only want repeat customers**, or are one-time buyers included
  too? (Gamma-Gamma typically excludes customers with 0 repeat purchases
  since there's no variance to estimate from a single order.)
- **Business context**: is this subscription-like (regular cadence) or
  non-contractual (retail, ad hoc purchases)? These models assume
  *non-contractual* repeat purchasing (e.g., ecommerce), not subscriptions.

Once you have (or can generate/pull) transaction data in that three-column
shape, we can build the full pipeline: fit BG/NBD → fit Gamma-Gamma →
generate CLV predictions.
"""

README_ZH = """
### 使用 BG/NBD + Gamma-Gamma 模型的客户终身价值分析

这是一个网页仪表盘，使用两个成熟的市场营销统计模型，预测每位客户未来能为
你带来多少价值。上传交易历史记录，即可获得按预测未来价值排序的客户名单。

## 这个工具具体做什么

**BG/NBD（Beta几何/负二项分布模型）**
分析每位客户*何时*以及*多频繁*购买，并预测：
- 未来一段时期内他们还会购买多少次
- 他们仍是活跃客户的概率（零售业没有"取消订阅"按钮，所以是否流失
  需要通过模型推断）

**Gamma-Gamma 模型**
分析客户*每次消费多少钱*，并预测他们未来的平均订单金额。之所以与
BG/NBD 分开建模，是因为假设购买频率与订单金额是相互独立的。

**综合结果：** 预测未来购买次数 × 预测平均订单金额 × 折现率 =
每位客户的预测客户终身价值（CLV）。

这些模型适用于**非合同制、重复消费型的业务**（电商、零售、餐饮等）——
不适用于订阅制业务，因为订阅制的流失是有明确记录的，更适合用其他模型。

## 你需要准备什么数据

只需要交易层面的记录，每笔订单一行：

| customer_id（客户ID） | order_date（订单日期） | order_value（订单金额） |
|---|---|---|
| 1001 | 2024-01-15 | 45.00 |
| 1001 | 2024-03-02 | 62.50 |
| 1002 | 2024-01-20 | 30.00 |

真的只需要这三列。系统会根据这些数据，为每位客户计算模型实际使用的
四个汇总指标：

- **frequency（频率）** — *重复*购买次数（总交易次数减一）
- **recency（最近一次购买）** — 首次购买与最近一次购买之间的时间间隔
- **T（观察期）** — 首次购买到"现在"（观察期结束）之间的时间
- **monetary_value（平均消费金额）** — 重复购买订单的平均金额

### 使用前需要考虑的几个问题

- **时间单位**：按天还是按周计算？会影响结果的可解读性。
- **你的数据回溯了多久**，是否希望预留最近一段时期的数据用于验证预测
  的准确性？
- **只统计重复购买的客户，还是也包含只买过一次的客户？**（Gamma-Gamma
  模型通常会排除0次重复购买的客户，因为单次订单无法估计其消费波动性。）
- **业务性质**：是订阅制（固定周期）还是非合同制（零售、随机消费）？
  这些模型假设的是*非合同制*的重复购买场景（如电商），不适用于订阅制业务。

准备好符合这三列格式的交易数据后（或可以生成/获取），我们就可以搭建
完整的分析流程：拟合 BG/NBD → 拟合 Gamma-Gamma → 生成 CLV 预测结果。
"""

TEXT = {
    "en": {
        "app_title": "Jason Peng's CLV Dashboard",
        "app_caption": "BG/NBD + Gamma-Gamma customer lifetime value modeling",
        "password_label": "Password",
        "password_error": "Incorrect password.",
        "sidebar_title": "Jason Peng's CLV Dashboard",
        "sidebar_caption": "BG/NBD + Gamma-Gamma",
        "data_source_label": "Data source",
        "data_source_options": ["Upload CSV", "Use synthetic sample data"],
        "upload_label": "Transactions CSV",
        "upload_help": "Must contain columns: customer_id, order_date, order_value",
        "synthetic_slider_label": "Synthetic customers",
        "time_unit_label": "Time unit",
        "time_unit_help": "D = days, W = weeks. Use weeks for very high-frequency purchasing.",
        "clv_horizon_label": "CLV horizon (months)",
        "discount_rate_label": "Monthly discount rate",
        "penalizer_label": "Penalizer coefficient (regularization)",
        "currency_label": "Your data's currency",
        "currency_options": ["USD ($)", "RMB (¥)"],
        "currency_help": "What currency are the order_value amounts in your uploaded CSV?",
        "exchange_rate_label": "Exchange rate (1 USD = ? RMB)",
        "exchange_rate_help": "Used to show CLV in both currencies. Update to the current rate as "
                               "needed (around ¥7.1-7.2 per USD as of mid-2026).",
        "run_button": "Run models",
        "main_caption": "BG/NBD (purchase frequency & churn) + Gamma-Gamma (spend) → CLV",
        "readme": README_EN,
        "info_upload_prompt": "Upload a transactions CSV in the sidebar, or switch to synthetic "
                               "sample data, to get started. Required columns: **customer_id**, "
                               "**order_date**, **order_value**.",
        "preview_expander": "Preview uploaded data",
        "preview_caption": "{n_txn:,} transactions, {n_cust:,} unique customers, date range "
                            "{start} → {end}",
        "info_adjust_settings": "Adjust settings in the sidebar, then click **Run models**.",
        "spinner_text": "Building RFM summary and fitting models...",
        "success_text": "Models fit successfully.",
        "kpi_customers": "Customers",
        "kpi_repeat": "Repeat customers",
        "kpi_repeat_pct": "{pct:.0f}% of total",
        "kpi_avg_usd": "Avg predicted {h}mo CLV (USD)",
        "kpi_avg_rmb": "Avg predicted {h}mo CLV (RMB)",
        "kpi_total_usd": "Total predicted CLV (USD)",
        "tab_overview": "Model overview",
        "tab_diag": "Diagnostics",
        "tab_table": "CLV table",
        "bgnbd_params": "BG/NBD parameters",
        "bgnbd_caption": "r, alpha describe the population's purchase-rate (Gamma) distribution. "
                          "a, b describe the population's churn-probability (Beta) distribution.",
        "gg_params": "Gamma-Gamma parameters",
        "corr_metric_label": "Frequency ↔ monetary value correlation",
        "corr_ok": "OK (near 0)",
        "corr_bad": "Check assumption",
        "gg_caption": "Gamma-Gamma assumes spend is independent of purchase frequency. Values "
                       "well above ~0.3 in magnitude mean this assumption may not hold well.",
        "top15_header": "Top 15 customers by predicted {h}-month CLV",
        "diag_caption": "Calibration/holdout validates predictions against real held-out "
                         "purchases. The other plots show how the fitted model behaves across "
                         "the frequency/recency space.",
        "cal_holdout_title": "**Calibration vs. holdout**",
        "cal_holdout_warning": "Couldn't build calibration/holdout split (often means the date "
                                "range is too short for the chosen CLV horizon): {e}",
        "freq_rec_title": "**Frequency / recency matrix**",
        "prob_alive_title": "**Probability alive matrix**",
        "period_txn_title": "**Model fit vs. actual (period transactions)**",
        "full_table_header": "Full CLV table",
        "filter_by": "Filter by",
        "min_clv_filter": "Minimum CLV filter ({cur})",
        "table_caption": "Showing {n_filtered:,} of {n_total:,} customers · exchange rate used: "
                          "1 USD = ¥{rate:.2f}",
        "download_button": "Download CLV table as CSV (both currencies)",
        "footer": "Built by Jason Peng · BG/NBD + Gamma-Gamma CLV modeling",
        "language_label": "Language / 语言",
    },
    "zh": {
        "app_title": "Jason Peng 的 CLV 客户终身价值仪表盘",
        "app_caption": "基于 BG/NBD + Gamma-Gamma 模型的客户终身价值分析",
        "password_label": "密码",
        "password_error": "密码错误。",
        "sidebar_title": "Jason Peng 的 CLV 仪表盘",
        "sidebar_caption": "BG/NBD + Gamma-Gamma",
        "data_source_label": "数据来源",
        "data_source_options": ["上传 CSV 文件", "使用模拟示例数据"],
        "upload_label": "交易记录 CSV 文件",
        "upload_help": "文件必须包含以下列：customer_id（客户ID）、order_date（订单日期）、"
                        "order_value（订单金额）",
        "synthetic_slider_label": "模拟客户数量",
        "time_unit_label": "时间单位",
        "time_unit_help": "D = 天，W = 周。若客户购买频率非常高（如每周多次），建议使用周。",
        "clv_horizon_label": "CLV 预测周期（月）",
        "discount_rate_label": "月折现率",
        "penalizer_label": "正则化系数（惩罚项）",
        "currency_label": "你的数据使用的货币",
        "currency_options": ["美元 USD ($)", "人民币 RMB (¥)"],
        "currency_help": "你上传的CSV文件中，order_value 使用的是什么货币？",
        "exchange_rate_label": "汇率（1 美元 = ? 人民币）",
        "exchange_rate_help": "用于同时显示两种货币的CLV结果。请根据当前汇率更新"
                               "（截至2026年年中约为7.1-7.2左右）。",
        "run_button": "运行模型",
        "main_caption": "BG/NBD（购买频率与流失）+ Gamma-Gamma（消费金额）→ CLV",
        "readme": README_ZH,
        "info_upload_prompt": "请在侧边栏上传交易记录 CSV 文件，或切换为使用模拟示例数据来开始。"
                               "必需的列：**customer_id**、**order_date**、**order_value**。",
        "preview_expander": "预览已上传的数据",
        "preview_caption": "共 {n_txn:,} 笔交易，{n_cust:,} 位独立客户，日期范围 {start} → {end}",
        "info_adjust_settings": "在侧边栏调整设置后，点击 **运行模型**。",
        "spinner_text": "正在构建 RFM 汇总并拟合模型……",
        "success_text": "模型拟合成功。",
        "kpi_customers": "客户总数",
        "kpi_repeat": "重复购买客户数",
        "kpi_repeat_pct": "占总数的 {pct:.0f}%",
        "kpi_avg_usd": "预测{h}个月平均 CLV（美元）",
        "kpi_avg_rmb": "预测{h}个月平均 CLV（人民币）",
        "kpi_total_usd": "预测总 CLV（美元）",
        "tab_overview": "模型概览",
        "tab_diag": "诊断分析",
        "tab_table": "CLV 数据表",
        "bgnbd_params": "BG/NBD 模型参数",
        "bgnbd_caption": "r、alpha 描述了整体客户群购买频率（Gamma分布）的特征；"
                          "a、b 描述了整体客户群流失概率（Beta分布）的特征。",
        "gg_params": "Gamma-Gamma 模型参数",
        "corr_metric_label": "购买频率与消费金额的相关性",
        "corr_ok": "正常（接近0）",
        "corr_bad": "请检查该假设",
        "gg_caption": "Gamma-Gamma 模型假设消费金额与购买频率相互独立。若该数值的绝对值"
                       "明显高于约0.3，说明这一假设可能不太成立。",
        "top15_header": "按预测{h}个月 CLV 排名前15的客户",
        "diag_caption": "校准/留出验证会用真实的留出数据检验预测的准确性。其余图表展示了"
                         "拟合模型在不同购买频率/最近购买时间组合下的表现。",
        "cal_holdout_title": "**校准 vs. 留出验证**",
        "cal_holdout_warning": "无法构建校准/留出数据集（通常是因为所选 CLV 预测周期相对"
                                "数据的时间跨度太长）：{e}",
        "freq_rec_title": "**购买频率 / 最近购买时间矩阵**",
        "prob_alive_title": "**客户存活概率矩阵**",
        "period_txn_title": "**模型拟合效果对比（周期交易数）**",
        "full_table_header": "完整 CLV 数据表",
        "filter_by": "筛选方式",
        "min_clv_filter": "最低 CLV 筛选（{cur}）",
        "table_caption": "显示 {n_total:,} 位客户中的 {n_filtered:,} 位 · 使用汇率："
                          "1 美元 = ¥{rate:.2f}",
        "download_button": "下载 CLV 数据表（含两种货币）",
        "footer": "由 Jason Peng 制作 · BG/NBD + Gamma-Gamma 客户终身价值模型",
        "language_label": "Language / 语言",
    },
}

# ----------------------------------------------------------------------
# Language toggle (must run before the password gate, so it also applies
# to the login screen)
# ----------------------------------------------------------------------

if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

_lang_col1, _lang_col2 = st.columns([5, 1])
with _lang_col2:
    _choice = st.selectbox(
        TEXT[st.session_state["lang"]]["language_label"],
        ["English", "中文"],
        index=0 if st.session_state["lang"] == "en" else 1,
        label_visibility="collapsed",
    )
    st.session_state["lang"] = "en" if _choice == "English" else "zh"

lang = st.session_state["lang"]
T = TEXT[lang]

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

    st.title(T["app_title"])
    st.caption(T["app_caption"])
    st.text_input(T["password_label"], type="password", key="password_input",
                  on_change=password_entered)
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error(T["password_error"])
    st.image("assets/password_banner.jpg", use_container_width=True)
    return False


if not check_password():
    st.stop()

# Hide the "Deploy" button and any GitHub icon/badge (shown automatically
# when deployed from a public repo on Streamlit Cloud), while leaving the
# hamburger menu (top-right "..." menu) visible and functional.
st.markdown(
    """
    <style>
    .stAppDeployButton {display: none;}
    [data-testid="stToolbarActions"] {display: none;}
    a[href*="github.com"] {display: none;}
    div[class*="viewerBadge"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

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

st.sidebar.title(T["sidebar_title"])
st.sidebar.caption(T["sidebar_caption"])

data_source = st.sidebar.radio(T["data_source_label"], T["data_source_options"], index=0)

if data_source == T["data_source_options"][0]:
    uploaded = st.sidebar.file_uploader(T["upload_label"], type=["csv"], help=T["upload_help"])
    raw_df = None
    if uploaded is not None:
        try:
            raw_df = load_transactions(uploaded)
        except Exception as e:
            st.sidebar.error(str(e))
else:
    n_cust = st.sidebar.slider(T["synthetic_slider_label"], 500, 10000, 3000, step=500)
    raw_df = make_synthetic_data(n_cust)

st.sidebar.divider()
time_unit_display = st.sidebar.selectbox(T["time_unit_label"], ["D", "W"], index=0,
                                          help=T["time_unit_help"])
time_unit = time_unit_display
clv_horizon = st.sidebar.number_input(T["clv_horizon_label"], min_value=1, max_value=60, value=12)
discount_rate = st.sidebar.number_input(T["discount_rate_label"], min_value=0.0, max_value=0.2,
                                         value=0.01, step=0.005, format="%.3f")
penalizer = st.sidebar.number_input(T["penalizer_label"], min_value=0.0,
                                     max_value=1.0, value=0.001, step=0.001, format="%.3f")

st.sidebar.divider()
input_currency = st.sidebar.selectbox(T["currency_label"], T["currency_options"], index=0,
                                       help=T["currency_help"])
exchange_rate = st.sidebar.number_input(
    T["exchange_rate_label"], min_value=0.01, value=7.15, step=0.01, format="%.2f",
    help=T["exchange_rate_help"],
)
is_usd_input = input_currency == T["currency_options"][0]

run_btn = st.sidebar.button(T["run_button"], type="primary", use_container_width=True,
                             disabled=raw_df is None)

# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------

st.title(T["app_title"])
st.caption(T["main_caption"])

st.image("assets/banner.jpg", use_container_width=True)
st.markdown(T["readme"])
st.divider()

if raw_df is None:
    st.info(T["info_upload_prompt"])
    st.stop()

with st.expander(T["preview_expander"], expanded=False):
    st.write(T["preview_caption"].format(
        n_txn=len(raw_df), n_cust=raw_df["customer_id"].nunique(),
        start=raw_df["order_date"].min().date(), end=raw_df["order_date"].max().date(),
    ))
    st.dataframe(raw_df.head(20), use_container_width=True)

if not run_btn:
    st.info(T["info_adjust_settings"])
    st.stop()

# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

with st.spinner(T["spinner_text"]):
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

    # Convert CLV into both currencies based on which currency the input data is in.
    if is_usd_input:
        summary["clv_usd"] = summary["clv"]
        summary["clv_rmb"] = summary["clv"] * exchange_rate
    else:
        summary["clv_rmb"] = summary["clv"]
        summary["clv_usd"] = summary["clv"] / exchange_rate

    summary = summary.sort_values("clv", ascending=False)

st.success(T["success_text"])

# ----------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric(T["kpi_customers"], f"{len(summary):,}")
k2.metric(T["kpi_repeat"], f"{len(repeat):,}",
          T["kpi_repeat_pct"].format(pct=len(repeat) / len(summary) * 100))
k3.metric(T["kpi_avg_usd"].format(h=clv_horizon), f"${summary['clv_usd'].mean():,.2f}")
k4.metric(T["kpi_avg_rmb"].format(h=clv_horizon), f"¥{summary['clv_rmb'].mean():,.2f}")
k5.metric(T["kpi_total_usd"], f"${summary['clv_usd'].sum():,.0f}")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

tab_overview, tab_diag, tab_table = st.tabs([T["tab_overview"], T["tab_diag"], T["tab_table"]])
with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(T["bgnbd_params"])
        st.dataframe(bgf.summary, use_container_width=True)
        st.caption(T["bgnbd_caption"])
    with col2:
        st.subheader(T["gg_params"])
        st.dataframe(ggf.summary, use_container_width=True)
        corr_ok = abs(fm_corr) < 0.3 if not np.isnan(fm_corr) else True
        st.metric(T["corr_metric_label"], f"{fm_corr:.3f}",
                  delta=T["corr_ok"] if corr_ok else T["corr_bad"],
                  delta_color="normal" if corr_ok else "inverse")
        st.caption(T["gg_caption"])

    st.subheader(T["top15_header"].format(h=clv_horizon))
    st.dataframe(
        summary.head(15)[["frequency", "recency", "T", "monetary_value",
                           "predicted_purchases", "prob_alive", "clv_usd", "clv_rmb"]]
        .style.format({"monetary_value": "${:,.2f}", "clv_usd": "${:,.2f}", "clv_rmb": "¥{:,.2f}",
                        "predicted_purchases": "{:.2f}", "prob_alive": "{:.1%}"}),
        use_container_width=True,
    )

with tab_diag:
    st.caption(T["diag_caption"])
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(T["cal_holdout_title"])
        try:
            calibration_end = observation_end - pd.Timedelta(
                days=int(horizon_periods) if time_unit == "D" else int(horizon_periods * 7)
            )
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
            st.warning(T["cal_holdout_warning"].format(e=e))

    with c2:
        st.markdown(T["freq_rec_title"])
        fig = fig_from_plotter(plot_frequency_recency_matrix, bgf)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(T["prob_alive_title"])
        fig = fig_from_plotter(plot_probability_alive_matrix, bgf)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c4:
        st.markdown(T["period_txn_title"])
        fig = fig_from_plotter(plot_period_transactions, bgf)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with tab_table:
    st.subheader(T["full_table_header"])

    filter_currency = st.radio(T["filter_by"], T["currency_options"], index=0, horizontal=True)
    filtering_usd = filter_currency == T["currency_options"][0]
    if filtering_usd:
        max_val = float(summary["clv_usd"].max())
        min_clv = st.slider(T["min_clv_filter"].format(cur="USD"), 0.0, max_val, 0.0)
        filtered = summary[summary["clv_usd"] >= min_clv]
    else:
        max_val = float(summary["clv_rmb"].max())
        min_clv = st.slider(T["min_clv_filter"].format(cur="RMB"), 0.0, max_val, 0.0)
        filtered = summary[summary["clv_rmb"] >= min_clv]

    st.caption(T["table_caption"].format(
        n_filtered=len(filtered), n_total=len(summary), rate=exchange_rate
    ))
    st.dataframe(
        filtered.style.format({"clv_usd": "${:,.2f}", "clv_rmb": "¥{:,.2f}",
                                "monetary_value": "{:,.2f}", "predicted_avg_value": "{:,.2f}",
                                "predicted_purchases": "{:.2f}", "prob_alive": "{:.1%}"}),
        use_container_width=True, height=450,
    )

    csv_buf = io.StringIO()
    filtered.to_csv(csv_buf)
    st.download_button(
        T["download_button"],
        data=csv_buf.getvalue(),
        file_name="clv_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(T["footer"])

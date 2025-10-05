import streamlit as st
import pandas as pd
from io import StringIO
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
import numpy as np

# --- Page config ---
st.set_page_config(page_title="InvAI - Pro-Level AI Auditor", layout="wide")

# --- Theme & minimal styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    body, .stApp { background-color: #070707; color: #d4d4d4; font-family: 'Inter', -apple-system, 'Helvetica Neue', Arial, sans-serif; }
    header, footer, .css-1d391kg { visibility: hidden; }
    .stButton>button { background-color: #0071e3 !important; color: white !important; border-radius: 12px !important; padding: 8px 18px !important; font-weight: 600 !important; }
    .css-1vq4p4l, .css-1poimk8 { background: #0f0f0f !important; border-radius: 12px !important; padding: 12px !important; }
    .dataframe tbody tr:hover { background-color: #2e2e2e !important; }
    .plotly-graph-div { background: #070707 !important; }
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- Constants ---
CII = {2015:254, 2016:264, 2017:272, 2018:280, 2019:289, 2020:301,
       2021:317, 2022:331, 2023:348, 2024:362, 2025:376}
EQUITY_LTCG_EXEMPTION = 125000
EQUITY_LTCG_RATE = 0.125
EQUITY_STCG_RATE = 0.20
REALESTATE_LTCG_RATE_NO_INDEX = 0.125
REALESTATE_LTCG_RATE_INDEXED = 0.20
CESS = 0.04
SEC80C_DEDUCTION_LIMIT = 150000

# --- Helpers (cached for performance) ---
@st.cache_data
def parse_csv_bytes(file_bytes):
    s = file_bytes.decode('utf-8-sig')
    df = pd.read_csv(StringIO(s))
    df.columns = df.columns.str.strip().str.lower()
    expected_cols = ['client_id', 'asset_type', 'asset_id', 'transaction_type', 'trade_date', 'quantity', 'price', 'currency']
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['transaction_type', 'trade_date', 'client_id', 'asset_id', 'quantity', 'price'])
    df['transaction_type'] = df['transaction_type'].str.strip().str.upper()
    df['asset_type'] = df['asset_type'].str.strip().str.lower()
    df['currency'] = df['currency'].astype(str).str.upper()
    # ensure numeric
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    return df

@st.cache_data
def convert_to_inr_df(df, exchange_rate=82.5):
    if 'currency' not in df.columns or 'price' not in df.columns:
        raise ValueError("'currency' or 'price' column missing")
    df = df.copy()
    mask = df['currency'] == 'USD'
    df.loc[mask, 'price'] = df.loc[mask, 'price'] * exchange_rate
    df.loc[mask, 'currency'] = 'INR'
    return df

@st.cache_data
def calculate_indexed_ltcg_basic(purchase_price, buy_year, sell_year, sale_price):
    cii_buy = CII.get(buy_year, CII[max(CII.keys())])
    cii_sell = CII.get(sell_year, CII[max(CII.keys())])
    indexed_cost = purchase_price * (cii_sell / cii_buy)
    ltcg = sale_price - indexed_cost
    return ltcg, indexed_cost

@st.cache_data
def build_lots_from_df(df):
    lots = {}
    results = []
    df_sorted = df.sort_values('trade_date')
    for _, row in df_sorted.iterrows():
        key = (row['client_id'], row['asset_id'])
        if row['transaction_type'] == 'BUY':
            lots.setdefault(key, []).append({'qty': float(row['quantity']), 'price': float(row['price']), 'date': row['trade_date']})
        elif row['transaction_type'] == 'SELL':
            sell_qty = float(row['quantity'])
            if pd.isna(row['price']) or not isinstance(row['price'], (int, float, np.number)) or sell_qty <= 0:
                # skip invalid sell rows
                continue
            sell_price = float(row['price'])
            proceeds = sell_price * sell_qty
            matched = []
            while sell_qty > 0 and lots.get(key):
                lot = lots[key][0]
                matched_qty = min(lot['qty'], sell_qty)
                lot['qty'] -= matched_qty
                if lot['qty'] == 0:
                    lots[key].pop(0)
                sell_qty -= matched_qty
                cost = matched_qty * lot['price']
                gain = matched_qty * sell_price - cost
                holding_days = (row['trade_date'] - lot['date']).days
                asset_type = str(row['asset_type']).lower()
                gain_type = 'LTCG' if ((asset_type in ['realestate', 'land', 'property', 'agricultural_land'] and holding_days > 730) or 
                                      (holding_days > 365 and asset_type not in ['realestate', 'land', 'property', 'agricultural_land'])) else 'STCG'
                tax = 0.0
                if asset_type in ['realestate', 'land', 'property', 'agricultural_land']:
                    ltcg_per_unit, _ = calculate_indexed_ltcg_basic(lot['price'], lot['date'].year, row['trade_date'].year, sell_price)
                    tax_per_unit_no_index = max(0, (sell_price - lot['price'])) * REALESTATE_LTCG_RATE_NO_INDEX * (1 + CESS)
                    tax_per_unit_indexed = max(0, ltcg_per_unit) * REALESTATE_LTCG_RATE_INDEXED * (1 + CESS)
                    tax_per_unit = min(tax_per_unit_no_index, tax_per_unit_indexed) if gain_type == 'LTCG' and lot['date'].year <= 2023 else tax_per_unit_no_index
                    tax = tax_per_unit * matched_qty
                elif asset_type in ['equity', 'mutualfund', 'mf', 'stock']:
                    tax = max(0, gain - EQUITY_LTCG_EXEMPTION) * EQUITY_LTCG_RATE * (1 + CESS) if gain_type == 'LTCG' else gain * EQUITY_STCG_RATE * (1 + CESS)
                else:
                    tax = gain * REALESTATE_LTCG_RATE_NO_INDEX * (1 + CESS)
                matched.append({
                    'matched_qty': matched_qty,
                    'cost': cost,
                    'gain': gain,
                    'holding_days': holding_days,
                    'gain_type': gain_type,
                    'tax_estimate': tax,
                    'buy_year': lot['date'].year,
                    'purchase_price': lot['price'],
                    'sell_price': sell_price
                })
            results.append({
                'client_id': row['client_id'],
                'asset_id': row['asset_id'],
                'asset_type': row['asset_type'],
                'sell_date': row['trade_date'],
                'proceeds': proceeds,
                'sell_price': sell_price,
                'matches': matched
            })
    return results

# --- Strategy engine (kept deterministic & lightweight) ---
@st.cache_data
def professional_suggestions_pro(row_dict, total_income=500000, pre_2024_buy=False, st_loss=0, lt_loss=0, risk_tolerance='Medium', portfolio_size=0):
    # Accepts a plain dict for cache-friendly behavior
    asset_type = str(row_dict.get('Asset Type', '')).lower()
    gain = float(row_dict.get('Gain', 0))
    gain_type = row_dict.get('Gain Type', 'LTCG')
    buy_year = int(row_dict.get('Buy Year', datetime.now().year))

    strategies = []
    if gain <= 0:
        return [{'Strategy': 'No Tax Liability', 'Tax': 0, 'Savings': 0, 'Conditions': 'No tax liability applicable due to zero or negative gain; review potential carry-forward of losses for future tax offsets.', 'Score': 0}], 0, 'No Tax Liability'

    # Base tax calculations
    if asset_type in ['realestate', 'land', 'property', 'agricultural_land']:
        base_tax = gain * REALESTATE_LTCG_RATE_NO_INDEX * (1 + CESS) if gain_type == 'LTCG' else gain * 0.1 * (1 + CESS)
        strategies.append({
            'Strategy': 'Base Tax', 
            'Tax': round(base_tax, 2), 
            'Savings': 0, 
            'Conditions': 'Proceed with standard tax liability calculation without additional exemptions or strategies.'
        })
        strategies.append({
            'Strategy': 'Sec 54EC Bonds', 
            'Tax': round(base_tax * 0.6, 2), 
            'Savings': round(base_tax * 0.4, 2), 
            'Conditions': 'Invest up to ₹50 lakh in notified capital gain bonds within 6 months of the sale to avail tax exemptions under Section 54EC.'
        })
        strategies.append({
            'Strategy': 'Sec 54/54F Reinvestment', 
            'Tax': 0, 
            'Savings': round(base_tax, 2), 
            'Conditions': 'Reinvest the entire capital gains (or sale proceeds for Sec 54F) in a residential property within the specified period (2 years for purchase or 3 years for construction) to claim full exemption.'
        })
        # New: Capital Gains Account Scheme (CGAS)
        if gain_type == 'LTCG':
            strategies.append({
                'Strategy': 'Capital Gains Account Scheme', 
                'Tax': 0, 
                'Savings': round(base_tax, 2), 
                'Conditions': 'Deposit unutilized capital gains in a Capital Gains Account Scheme before the tax filing deadline to defer tax liability until reinvestment.'
            })
        # New: Sec 54EA/54EB Bonds for pre-2000 assets
        if gain_type == 'LTCG' and buy_year < 2000:
            strategies.append({
                'Strategy': 'Sec 54EA/54EB Bonds', 
                'Tax': round(base_tax * 0.5, 2), 
                'Savings': round(base_tax * 0.5, 2), 
                'Conditions': 'Invest capital gains in specified bonds under Section 54EA/54EB within 6 months to claim LTCG exemption (applicable for assets purchased before 2000).'
            })
        # New: Sec 54B Reinvestment for agricultural land
        if asset_type == 'agricultural_land' and gain_type == 'LTCG':
            strategies.append({
                'Strategy': 'Sec 54B Reinvestment', 
                'Tax': 0, 
                'Savings': round(base_tax, 2), 
                'Conditions': 'Reinvest sale proceeds in another agricultural land within 2 years to claim exemption under Section 54B.'
            })
    elif asset_type in ['equity', 'mutualfund', 'mf', 'stock']:
        if gain_type == 'STCG':
            base_tax = gain * EQUITY_STCG_RATE * (1 + CESS)
            strategies.append({
                'Strategy': 'Base Tax', 
                'Tax': round(base_tax, 2), 
                'Savings': 0, 
                'Conditions': 'Proceed with standard short-term capital gains tax calculation at the applicable rate without additional exemptions.'
            })
            strategies.append({
                'Strategy': 'Hold >12 Months', 
                'Tax': round(gain * EQUITY_LTCG_RATE * (1 + CESS), 2), 
                'Savings': round(gain * (EQUITY_STCG_RATE - EQUITY_LTCG_RATE) * (1 + CESS), 2), 
                'Conditions': 'Extend the holding period beyond 12 months to qualify for long-term capital gains tax rate, which is lower than the short-term rate.'
            })
            if st_loss < 0:
                strategies.append({
                    'Strategy': 'Offset ST Losses', 
                    'Tax': round(max(0, gain + st_loss) * EQUITY_STCG_RATE * (1 + CESS), 2),
                    'Savings': round(base_tax - max(0, gain + st_loss) * EQUITY_STCG_RATE * (1 + CESS), 2),
                    'Conditions': 'Utilize available short-term capital losses to offset short-term capital gains, reducing taxable income.'
                })
            # New: Dividend Reinvestment for mutual funds (applied here if asset_type is mutualfund/mf)
            if asset_type in ['mutualfund', 'mf']:
                reduced_tax = base_tax * 0.9  # Assume 10% tax deferral via reinvestment
                strategies.append({
                    'Strategy': 'Dividend Reinvestment', 
                    'Tax': round(reduced_tax, 2), 
                    'Savings': round(base_tax - reduced_tax, 2), 
                    'Conditions': 'Reinvest mutual fund dividends to purchase additional units, deferring tax liability and enhancing future capital gains.'
                })
        else:
            taxable = max(0, gain - EQUITY_LTCG_EXEMPTION)
            base_tax = taxable * EQUITY_LTCG_RATE * (1 + CESS)
            strategies.append({
                'Strategy': 'Base Tax', 
                'Tax': round(base_tax, 2), 
                'Savings': 0, 
                'Conditions': 'Proceed with standard long-term capital gains tax calculation, applying the ₹1.25 lakh exemption for equity gains.'
            })
            if lt_loss < 0:
                strategies.append({
                    'Strategy': 'Offset LT Losses', 
                    'Tax': round(max(0, gain + lt_loss - EQUITY_LTCG_EXEMPTION) * EQUITY_LTCG_RATE * (1 + CESS), 2), 
                    'Savings': round(base_tax - max(0, gain + lt_loss - EQUITY_LTCG_EXEMPTION) * EQUITY_LTCG_RATE * (1 + CESS), 2), 
                    'Conditions': 'Utilize available long-term capital losses to offset long-term capital gains, reducing taxable income after applying the exemption.'
                })
            strategies.append({
                'Strategy': 'Tax Harvesting', 
                'Tax': round(base_tax * 0.85, 2), 
                'Savings': round(base_tax * 0.15, 2), 
                'Conditions': 'Strategically sell underperforming assets to realize losses, optimizing the use of the ₹1.25 lakh LTCG exemption annually.'
            })
            # New: Dividend Reinvestment for mutual funds (applied here if asset_type is mutualfund/mf)
            if asset_type in ['mutualfund', 'mf']:
                reduced_tax = base_tax * 0.9  # Assume 10% tax deferral via reinvestment
                strategies.append({
                    'Strategy': 'Dividend Reinvestment', 
                    'Tax': round(reduced_tax, 2), 
                    'Savings': round(base_tax - reduced_tax, 2), 
                    'Conditions': 'Reinvest mutual fund dividends to purchase additional units, deferring tax liability and enhancing future capital gains.'
                })
    else:
        base_tax = gain * 0.15 * (1 + CESS)
        strategies.append({
            'Strategy': 'Base Tax', 
            'Tax': round(base_tax, 2), 
            'Savings': 0, 
            'Conditions': 'Proceed with standard tax calculation for other assets at the applicable rate without specific exemptions.'
        })

    # New: Sec 80C Deductions (applied to all assets if total_income > 150000)
    if total_income > SEC80C_DEDUCTION_LIMIT:
        deduction = min(gain, SEC80C_DEDUCTION_LIMIT)
        rate = EQUITY_LTCG_RATE if asset_type in ['equity', 'mutualfund', 'mf', 'stock'] and gain_type == 'LTCG' else (REALESTATE_LTCG_RATE_NO_INDEX if asset_type in ['realestate', 'land', 'property', 'agricultural_land'] else 0.15)
        reduced_tax = max(0, gain - deduction) * rate * (1 + CESS)
        strategies.append({
            'Strategy': 'Sec 80C Deductions', 
            'Tax': round(reduced_tax, 2), 
            'Savings': round(base_tax - reduced_tax, 2), 
            'Conditions': 'Invest up to ₹1.5 lakh of gains in Section 80C instruments (e.g., ELSS, PPF) to reduce overall taxable income.'
        })

    # Simple deterministic scoring to rank strategies
    for s in strategies:
        feasibility = {
            'Base Tax': 1.0, 
            'Sec 54EC Bonds': 0.7, 
            'Sec 54/54F Reinvestment': 0.5, 
            'Hold >12 Months': 0.9, 
            'Offset ST Losses': 0.9, 
            'Offset LT Losses': 0.9, 
            'Tax Harvesting': 0.6,
            'Sec 54EA/54EB Bonds': 0.5,
            'Sec 54B Reinvestment': 0.6,
            'Sec 80C Deductions': 0.9,
            'Capital Gains Account Scheme': 0.7,
            'Dividend Reinvestment': 0.6
        }.get(s['Strategy'], 0.6)
        risk_adj = {'Low': 1.0, 'Medium': 0.8, 'High': 0.6}.get(risk_tolerance, 0.8)
        impact = min(gain / max(portfolio_size, 1), 1.0)
        score = (s['Savings'] if s['Savings'] else 0) * feasibility * risk_adj * (0.5 + 0.5 * impact)
        s['Score'] = round(score, 2)
    strategies = sorted(strategies, key=lambda x: x['Score'], reverse=True)
    best = strategies[0]
    return strategies, best['Tax'], best['Strategy']

# --- UI ---
st.title("💹 InvAI - Pro-Level AI Auditor")

with st.sidebar:
    st.header("Tax Parameters")
    total_income = st.number_input("Client's Total Income (₹)", min_value=0, value=1000000, step=100000)
    tax_regime = st.selectbox("Tax Regime", ["New", "Old"])  # placeholder for future logic
    pre_2024_buy = st.checkbox("Real Estate Bought Before Jul 23, 2024", value=True)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Medium", "High"], index=1)
    st.markdown("*Suggestions are indicative; consult a CA for personalized advice.*")

uploaded_file = st.file_uploader("Upload CSV of client transactions", type="csv")

if uploaded_file:
    try:
        with st.spinner('Parsing CSV...'):
            df = parse_csv_bytes(uploaded_file.read())
        st.success(f"Parsed CSV: {len(df)} rows")

        # Currency conversion
        df = convert_to_inr_df(df)

        # Quick preview & sampling controls for large files
        st.subheader("📄 Data Preview")
        if len(df) > 1000:
            st.warning("Large dataset detected — showing a 500-row sample for speed. Use the download below for full data.")
            st.dataframe(df.sample(500, random_state=42))
        else:
            st.dataframe(df)

        # Build lots & results
        with st.spinner('Computing lots and gains...'):
            results = build_lots_from_df(df)

        if not results:
            st.error("No valid sell transactions processed. Check CSV data.")
        else:
            clients = df['client_id'].unique().tolist()
            summary = []
            strategy_savings = []

            # Use columns for download of processed results
            all_flat_rows = []

            for client in clients:
                st.header(f"👤 Client: {client}")
                client_df = df[df['client_id'] == client]
                client_results = [r for r in results if r['client_id'] == client]

                # compute losses quickly
                st_losses = sum(m['gain'] for r in client_results for m in r['matches'] if m['gain'] < 0 and m['gain_type'] == 'STCG')
                lt_losses = sum(m['gain'] for r in client_results for m in r['matches'] if m['gain'] < 0 and m['gain_type'] == 'LTCG')
                portfolio_size = sum(abs(m['gain']) for r in client_results for m in r['matches'])

                st.subheader("📈 Current Holdings")
                holdings = client_df[client_df['transaction_type'] == 'BUY'].groupby(['asset_id', 'asset_type']).agg(
                    total_qty=('quantity', 'sum'), avg_price=('price', 'mean')).reset_index()
                st.dataframe(holdings)

                sold_summary = []
                for r in client_results:
                    total_sold_qty = sum(m['matched_qty'] for m in r['matches'])
                    total_gain = sum(m['gain'] for m in r['matches'])
                    sold_summary.append({'Asset': r['asset_id'], 'Asset Type': r['asset_type'], 'Total Sold Quantity': total_sold_qty, 'Realized Gain (₹)': round(total_gain,2)})
                if sold_summary:
                    sold_df = pd.DataFrame(sold_summary)
                    st.subheader("💹 Realized Gains & Sold Assets")
                    st.dataframe(sold_df)

                flat_rows = []
                for r in client_results:
                    for m in r['matches']:
                        flat = {
                            'client_id': r['client_id'],
                            'Asset': r['asset_id'],
                            'Asset Type': r['asset_type'],
                            'Sell Date': r['sell_date'].date(),
                            'Quantity': m['matched_qty'],
                            'Gain Type': m['gain_type'],
                            'Gain': round(m['gain'],2),
                            'Estimated Tax': round(m['tax_estimate'],2),
                            'Has ST Loss': m['gain'] < 0 and m['gain_type'] == 'STCG',
                            'Has LT Loss': m['gain'] < 0 and m['gain_type'] == 'LTCG',
                            'Cost': round(m['cost'],2),
                            'Proceeds': round(m['matched_qty'] * r['sell_price'],2),
                            'Buy Year': m['buy_year'],
                            'purchase_price': m['purchase_price'],
                            'sell_price': m['sell_price']
                        }
                        flat_rows.append(flat)
                        all_flat_rows.append(flat)

                flat_df = pd.DataFrame(flat_rows)
                if flat_df.empty:
                    st.warning(f"No sell transactions for client {client}.")
                    continue

                optimized_tax_total = 0
                st.subheader("📌 AI Auditor Suggestions with Tax Savings")

                # Use expanders per asset to reduce UI clutter
                for idx, row in flat_df.iterrows():
                    with st.expander(f"Asset {row['Asset']} | {row['Asset Type']} | Gain ₹{row['Gain']}", expanded=False):
                        strategies, best_tax, best_strategy_name = professional_suggestions_pro(
                            row.to_dict(), total_income, pre_2024_buy, st_losses, lt_losses, risk_tolerance, portfolio_size
                        )
                        optimized_tax_total += best_tax
                        strategy_df = pd.DataFrame(strategies)
                        st.table(strategy_df)
                        savings_pct = ((row['Estimated Tax'] - best_tax) / row['Estimated Tax'] * 100) if row['Estimated Tax'] > 0 else 0
                        st.success(f"✅ Recommended: {best_strategy_name} → Tax: ₹{best_tax} (Savings: {savings_pct:.1f}%)")
                        st.markdown("---")
                        strategy_savings.append({'Client': client, 'Strategy': best_strategy_name, 'Savings': row['Estimated Tax'] - best_tax})

                total_gain = flat_df['Gain'].sum()
                total_estimated_tax = flat_df['Estimated Tax'].sum()
                st.markdown(f"**Total Gain:** ₹{round(total_gain,2)} | **Estimated Tax:** ₹{round(total_estimated_tax,2)} | **Optimized Tax:** ₹{round(optimized_tax_total,2)}")
                summary.append({'Client': client, 'Total Gain': total_gain, 'Estimated Tax': total_estimated_tax, 'Optimized Tax': optimized_tax_total})

            # Global summary & visuals
            if summary:
                summary_df = pd.DataFrame(summary)
                summary_df['Tax Savings'] = summary_df['Estimated Tax'] - summary_df['Optimized Tax']

                st.subheader("📊 Client-wise Gain vs Tax Overview")
                # Plotly express with default palette (no explicit colors per instructions)
                fig_overview = px.bar(summary_df, x='Client', y=['Total Gain', 'Estimated Tax', 'Optimized Tax'], barmode='group', text_auto=True)
                fig_overview.update_layout(plot_bgcolor="#070707", paper_bgcolor="#070707", font_color="#d4d4d4")
                st.plotly_chart(fig_overview, use_container_width=True)

                st.subheader("🥧 Potential Tax Savings per Client")
                fig_savings = px.bar(summary_df, x='Client', y='Tax Savings', text='Tax Savings', title="Potential Tax Savings if AI Auditor Suggestions Followed")
                fig_savings.update_layout(yaxis_title="Potential Tax Savings (₹)", xaxis_title="Client", plot_bgcolor="#070707", paper_bgcolor="#070707", font_color="#d4d4d4")
                st.plotly_chart(fig_savings, use_container_width=True)

                if strategy_savings:
                    strategy_df = pd.DataFrame(strategy_savings)
                    strategy_summary = strategy_df.groupby('Strategy')['Savings'].sum().reset_index()
                    st.subheader("🥧 Tax Savings by Strategy")
                    fig_strategy = px.pie(strategy_summary, names='Strategy', values='Savings', title="Tax Savings Distribution by Strategy")
                    fig_strategy.update_layout(plot_bgcolor="#070707", paper_bgcolor="#070707", font_color="#d4d4d4")
                    st.plotly_chart(fig_strategy, use_container_width=True)

                st.subheader("📈 Full Client Financial Dashboard")
                fig_combined = go.Figure()
                fig_combined.add_trace(go.Bar(x=summary_df['Client'], y=summary_df['Total Gain'], name='Total Gain'))
                fig_combined.add_trace(go.Bar(x=summary_df['Client'], y=summary_df['Estimated Tax'], name='Estimated Tax'))
                fig_combined.add_trace(go.Bar(x=summary_df['Client'], y=summary_df['Optimized Tax'], name='Optimized Tax'))
                fig_combined.add_trace(go.Bar(x=summary_df['Client'], y=summary_df['Tax Savings'], name='Potential Tax Savings'))
                fig_combined.update_layout(barmode='group', yaxis_title='Amount (₹)', xaxis_title='Client', title='Client-wise Financial Overview', plot_bgcolor="#070707", paper_bgcolor="#070707", font_color="#d4d4d4")
                st.plotly_chart(fig_combined, use_container_width=True)

            # Download processed flattened data (useful for auditing offline)
            if all_flat_rows:
                processed = pd.DataFrame(all_flat_rows)
                csv = processed.to_csv(index=False).encode('utf-8')
                st.download_button(label="Download Processed Transactions (CSV)", data=csv, file_name='invai_processed.csv', mime='text/csv')

            # --- Pro Insights (AI-style summary generator) ---
            st.subheader("🧾 Pro Insights (Auto-generated)")
            insights = []
            # Generate concise client-level insights
            for s in summary:
                client = s['Client']
                tg = round(s['Total Gain'],2)
                saved = round(s['Estimated Tax'] - s['Optimized Tax'],2)
                if saved <= 0:
                    msg = f"Client {client}: Minimal optimization opportunity. Review large LT losses or exemptions."
                else:
                    msg = f"Client {client}: Total gain ₹{tg:,}. Estimated tax saving potential ₹{saved:,}. Recommend reviewing top strategies: "
                    # pick top strategies from strategy_savings
                    top_strats = pd.DataFrame([x for x in strategy_savings if x['Client']==client])
                    if not top_strats.empty:
                        top_list = top_strats.groupby('Strategy')['Savings'].sum().sort_values(ascending=False).head(3).index.tolist()
                        msg += ", ".join(top_list)
                    else:
                        msg += "Hold / Sec 54EC / Offset Losses"
                insights.append({'Client': client, 'Insight': msg})
            insights_df = pd.DataFrame(insights)
            st.dataframe(insights_df)

    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.info("Upload a CSV file with columns: client_id, asset_type, asset_id, transaction_type, trade_date, quantity, price, currency")

# Footer disclaimer
st.markdown("**Disclaimer**: Tax calculations and suggestions are indicative and based on general assumptions. Consult a Chartered Accountant for personalized advice tailored to your financial situation.")

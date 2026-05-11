from flask import Flask
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Important! Stops matplotlib opening desktop windows
import matplotlib.pyplot as plt
from scipy.stats import norm
import base64
import io

app = Flask(__name__)

def generate_charts():
    # ── 1. Download 5 years of AAPL data ──────────────────────
    aapl = yf.download("AAPL", period="5y", auto_adjust=True)
    prices = aapl["Close"].squeeze()

    # ── 2. Daily returns ──────────────────────────────────────
    returns = prices.pct_change().dropna()

    # ── 3. Cumulative return ──────────────────────────────────
    cum_return = (1 + returns).cumprod() - 1

    # ── 4. Key metrics ────────────────────────────────────────
    volatility = returns.std() * np.sqrt(252) * 100
    sharpe     = (returns.mean() / returns.std()) * np.sqrt(252)
    total_ret  = cum_return.iloc[-1] * 100
    best_day   = returns.max() * 100
    worst_day  = returns.min() * 100
    start_price = round(float(prices.iloc[0]), 2)
    end_price   = round(float(prices.iloc[-1]), 2)

    # ── 5. Chart 1: Cumulative Return Curve ───────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(cum_return.index, cum_return * 100, color="#3b82f6", linewidth=1.5)
    ax1.fill_between(cum_return.index, cum_return * 100, 0,
                     where=(cum_return >= 0), alpha=0.1, color="#3b82f6")
    ax1.fill_between(cum_return.index, cum_return * 100, 0,
                     where=(cum_return < 0),  alpha=0.1, color="#ef4444")
    ax1.set_title("AAPL Cumulative Return (5 Years)", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Cumulative Return (%)")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert Chart 1 to base64 image
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=150)
    buf1.seek(0)
    chart1 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close(fig1)

    # ── 6. Chart 2: Histogram + Normal Distribution ───────────
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    daily_pct = returns * 100
    ax2.hist(daily_pct, bins=80, density=True,
             color="#3b82f6", alpha=0.7, label="Daily Returns")

    mu  = float(daily_pct.mean())
    std = float(daily_pct.std())
    x   = np.linspace(mu - 4*std, mu + 4*std, 300)
    ax2.plot(x, norm.pdf(x, mu, std), color="#f97316",
             linewidth=2.5, label="Normal Distribution")
    ax2.axvline(mu, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean: {mu:.3f}%")

    ax2.set_title("Distribution of AAPL Daily Returns", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Daily Return (%)")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Convert Chart 2 to base64 image
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=150)
    buf2.seek(0)
    chart2 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close(fig2)

    return {
        "chart1": chart1,
        "chart2": chart2,
        "volatility": round(float(volatility), 2),
        "sharpe":     round(float(sharpe), 2),
        "total_ret":  round(float(total_ret), 2),
        "best_day":   round(float(best_day), 2),
        "worst_day":  round(float(worst_day), 2),
        "start_price": start_price,
        "end_price":   end_price,
    }

@app.route("/")
def dashboard():
    data = generate_charts()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AAPL Stock Analysis</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                margin: 0;
                padding: 20px;
            }}
            h1 {{
                text-align: center;
                color: #3b82f6;
                font-size: 28px;
                margin-bottom: 5px;
            }}
            p.sub {{
                text-align: center;
                color: #94a3b8;
                margin-bottom: 30px;
            }}
            .cards {{
                display: flex;
                gap: 16px;
                justify-content: center;
                flex-wrap: wrap;
                margin-bottom: 30px;
            }}
            .card {{
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 12px;
                padding: 20px 30px;
                text-align: center;
                min-width: 160px;
            }}
            .card .label {{
                font-size: 13px;
                color: #94a3b8;
                margin-bottom: 8px;
            }}
            .card .value {{
                font-size: 24px;
                font-weight: bold;
            }}
            .green {{ color: #22c55e; }}
            .red   {{ color: #ef4444; }}
            .blue  {{ color: #3b82f6; }}
            .chart-box {{
                background: #1e293b;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 24px;
                text-align: center;
            }}
            .chart-box img {{
                width: 100%;
                max-width: 1000px;
                border-radius: 8px;
            }}
            table {{
                width: 100%;
                max-width: 600px;
                margin: 0 auto 30px auto;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px 20px;
                text-align: left;
                border-bottom: 1px solid #334155;
            }}
            tr:nth-child(even) {{ background: #1e293b; }}
            th {{ color: #94a3b8; font-size: 13px; }}
            footer {{
                text-align: center;
                color: #475569;
                font-size: 12px;
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>🍎 AAPL Stock Analysis</h1>
        <p class="sub">5-Year Performance Dashboard | Data from Yahoo Finance</p>

        <!-- Metric Cards -->
        <div class="cards">
            <div class="card">
                <div class="label">Total Return</div>
                <div class="value {'green' if data['total_ret'] > 0 else 'red'}">
                    {data['total_ret']}%
                </div>
            </div>
            <div class="card">
                <div class="label">Ann. Volatility</div>
                <div class="value blue">{data['volatility']}%</div>
            </div>
            <div class="card">
                <div class="label">Sharpe Ratio</div>
                <div class="value {'green' if data['sharpe'] > 1 else 'blue'}">{data['sharpe']}</div>
            </div>
            <div class="card">
                <div class="label">Best Day</div>
                <div class="value green">+{data['best_day']}%</div>
            </div>
            <div class="card">
                <div class="label">Worst Day</div>
                <div class="value red">{data['worst_day']}%</div>
            </div>
        </div>

        <!-- Chart 1 -->
        <div class="chart-box">
            <img src="data:image/png;base64,{data['chart1']}" alt="Cumulative Return">
        </div>

        <!-- Chart 2 -->
        <div class="chart-box">
            <img src="data:image/png;base64,{data['chart2']}" alt="Daily Returns Histogram">
        </div>

        <!-- Stats Table -->
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Start Price</td><td>${data['start_price']}</td></tr>
            <tr><td>End Price</td><td>${data['end_price']}</td></tr>
            <tr><td>Total Return</td><td>{data['total_ret']}%</td></tr>
            <tr><td>Annualized Volatility</td><td>{data['volatility']}%</td></tr>
            <tr><td>Sharpe Ratio</td><td>{data['sharpe']}</td></tr>
            <tr><td>Best Single Day</td><td>+{data['best_day']}%</td></tr>
            <tr><td>Worst Single Day</td><td>{data['worst_day']}%</td></tr>
        </table>

        <footer>Data: Yahoo Finance (AAPL) | Risk-free rate = 0%</footer>
    </body>
    </html>
    """
    return html

if __name__ == "__main__":
    app.run()
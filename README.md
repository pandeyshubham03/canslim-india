# CANSLIM India — Top 250 Dynamic Research Website

A polished Streamlit website for a Working-with-AI Accounting project. The UI is intentionally minimal and product-like: large typography, neutral surfaces, restrained blue interaction accents and explainable stock research rather than a cluttered trading terminal.

## Features

- Top-250-by-market-value universe
- Public Screener screen connector (best effort)
- Screener CSV upload for robust fundamentals
- Live / recent NSE price and volume history through `yfinance`
- NIFTY market-regime overlay
- CAN SLIM C/A/N/S/L/I factor engine
- Equal, literature and custom weighting
- Optional data-derived logistic-regression model from historical snapshots
- Sector strength ranking
- Best stock candidate in each sector
- Separate actionability / pivot / buy-zone proxy
- Full Top-250 leaderboard
- Stock deep-dive radar and evidence table
- Watchlist and CSV export
- Responsive, Apple-inspired light UI

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\\Scripts\\activate       # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Host free on Streamlit Community Cloud

1. Create a GitHub repo (for example `canslim-india`).
2. Upload `app.py`, `core.py`, `requirements.txt`, `.streamlit/config.toml` and the rest of this folder.
3. Open Streamlit Community Cloud.
4. Click **Create app**.
5. Choose your GitHub repository and branch.
6. Set the main file to `app.py`.
7. Deploy.

The app receives a public `*.streamlit.app` address. GitHub pushes redeploy automatically.

## Data strategy

Screener.in does not expose a public API. This project therefore uses two safe patterns:

- **Robust:** export a Screener screen CSV, upload it to the app, then merge with current price/volume history.
- **Best effort:** read a public top-250 Screener screen. This can break if page structure or access policies change.

For an academic submission, the recommended hosted setup is a periodically updated Screener CSV plus live market history. This avoids pretending a public API exists.

## Recommended Screener CSV columns

- Name
- NSE Code
- Industry
- Market Capitalization
- Current Price
- Qtr Profit Var %
- Qtr Sales Var %
- Sales growth 3Years
- Profit growth 3Years
- Return on equity
- Return on capital employed
- FII holding
- DII holding

Missing fields are tolerated. The dashboard shows **factor coverage** so low-information scores are not presented as equally reliable.

## MarketSmith methodology note

The website follows publicly described CAN SLIM concepts and adds transparent proxies for:

- earnings strength
- price leadership
- accumulation / demand
- industry strength
- pivot / breakout / buy-zone behaviour

It does **not** reproduce or claim to reproduce MarketSmith India's proprietary Master Score, exact ratings, chart-pattern engine or industry-group formula.

## Data-derived model template

Upload a historical CSV containing:

`C,A,N,S,L,I,forward_6m_excess_return`

Every row must contain only information known on the historical observation date. This is required to avoid look-ahead bias.

## Disclaimer

Academic research tool. Not personalized investment advice.

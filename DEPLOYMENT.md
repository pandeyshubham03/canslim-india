# Deployment checklist

## 1. GitHub

Create a new repository named `canslim-india` and upload the project files.

Recommended repo structure:

```
canslim-india/
├── app.py
├── core.py
├── requirements.txt
├── README.md
├── DEPLOYMENT.md
├── historical_training_template.csv
└── .streamlit/
    └── config.toml
```

## 2. Streamlit Community Cloud

- Sign in with GitHub.
- Create a new app.
- Repository: `YOUR_GITHUB_USERNAME/canslim-india`
- Branch: `main`
- Main file: `app.py`
- Deploy.

## 3. Recommended free-production setup

- Keep the app public.
- Keep any Screener export in the repo only if you are comfortable making that data public.
- Otherwise use the dashboard's CSV upload control per session.
- Current market prices/volume are refreshed with a cache interval to reduce load.
- Do not add passwords or API keys directly in code. Put secrets in Streamlit's Secrets UI if you later add a paid/official market-data API.

## 4. Free hosting caveats

Free services may sleep, rate-limit or restart. The dashboard is designed to recover using cached/demo data and user-uploaded Screener CSVs. For a classroom presentation, refresh the production app before presenting and keep a local copy available.

## 5. Custom domain later

For the course project, the free `streamlit.app` domain is enough. A branded custom domain can be added later by moving the same code to a platform that supports your preferred domain setup.

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

SCREENER_URL = "https://www.screener.in/screens/1341767/top-250-companies-by-market-cap/?order=desc&page={page}"
UA = {"User-Agent": "Mozilla/5.0 (CANSLIM-India-Academic-Dashboard/1.0)"}

LITERATURE_WEIGHTS = {"C": .22, "A": .18, "N": .10, "S": .15, "L": .25, "I": .10}
EQUAL_WEIGHTS = {k: 1/6 for k in "CANSLI"}


def _num(x):
    if x is None:
        return np.nan
    s = str(x).replace(",", "").replace("₹", "").replace("%", "").replace("Cr.", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else np.nan


def percentile(s: pd.Series, higher=True) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if x.notna().sum() < 2:
        return pd.Series(np.where(x.notna(), 50.0, np.nan), index=s.index)
    p = x.rank(pct=True, method="average") * 100
    return p if higher else 100 - p


def weighted_available(df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    out = []
    for _, row in df.iterrows():
        vals = [(row.get(k, np.nan), w) for k, w in weights.items()]
        vals = [(float(v), w) for v, w in vals if pd.notna(v)]
        if not vals:
            out.append(np.nan)
            continue
        denom = sum(w for _, w in vals)
        out.append(sum(v*w for v, w in vals)/denom)
    return pd.Series(out, index=df.index)


def synthesize_demo(n=250, seed=11):
    rng = np.random.default_rng(seed)
    sectors = ["Banks","IT","Pharma","Automobiles","Capital Goods","FMCG","Energy","Metals","Power","Financial Services","Telecom","Consumer","Infrastructure","Healthcare","Chemicals"]
    d = pd.DataFrame({
        "company": [f"Demo Company {i:03d}" for i in range(1,n+1)],
        "symbol": [f"DEMO{i:03d}" for i in range(1,n+1)],
        "sector": rng.choice(sectors, n),
        "market_cap_cr": np.sort(rng.lognormal(11.5,.8,n))[::-1],
        "price": rng.uniform(80,4200,n),
        "qtr_profit_yoy": rng.normal(22,30,n),
        "sales_yoy": rng.normal(15,18,n),
        "roce": np.clip(rng.normal(18,10,n),-10,80),
        "eps_yoy": rng.normal(25,30,n),
        "eps_acceleration": rng.normal(4,18,n),
        "opm_change": rng.normal(.8,3,n),
        "eps_cagr_3y": rng.normal(18,14,n),
        "sales_cagr_3y": rng.normal(14,10,n),
        "roe": np.clip(rng.normal(18,8,n),-5,65),
        "cfo_positive": rng.choice([0,1],n,p=[.12,.88]),
        "institutional_holding": np.clip(rng.normal(27,13,n),0,80),
        "institutional_change": rng.normal(.5,2,n),
        "return_3m": rng.normal(6,14,n),
        "return_6m": rng.normal(11,22,n),
        "return_12m": rng.normal(20,34,n),
        "distance_52w_high_pct": -np.abs(rng.normal(9,10,n)),
        "volume_ratio": np.clip(rng.lognormal(.08,.5,n),.25,5),
        "up_down_volume_ratio": np.clip(rng.lognormal(.08,.55,n),.2,6),
        "above_50dma": rng.choice([0,1],n,p=[.34,.66]),
        "above_200dma": rng.choice([0,1],n,p=[.28,.72]),
        "catalyst_score": np.clip(rng.normal(55,22,n),0,100),
    })
    # Seed a small set of obvious "leader" profiles so Demo Mode showcases the full UX.
    leaders = d.index[:10]
    d.loc[leaders, "eps_yoy"] = np.linspace(75, 48, len(leaders))
    d.loc[leaders, "qtr_profit_yoy"] = d.loc[leaders, "eps_yoy"]
    d.loc[leaders, "sales_yoy"] = np.linspace(42, 27, len(leaders))
    d.loc[leaders, "eps_acceleration"] = np.linspace(24, 11, len(leaders))
    d.loc[leaders, "eps_cagr_3y"] = np.linspace(38, 25, len(leaders))
    d.loc[leaders, "sales_cagr_3y"] = np.linspace(29, 20, len(leaders))
    d.loc[leaders, "roe"] = np.linspace(34, 24, len(leaders))
    d.loc[leaders, "institutional_holding"] = np.linspace(52, 36, len(leaders))
    d.loc[leaders, "institutional_change"] = np.linspace(4.2, 1.8, len(leaders))
    d.loc[leaders, "return_3m"] = np.linspace(31, 18, len(leaders))
    d.loc[leaders, "return_6m"] = np.linspace(58, 36, len(leaders))
    d.loc[leaders, "return_12m"] = np.linspace(96, 56, len(leaders))
    d.loc[leaders, "distance_52w_high_pct"] = -np.linspace(1, 6, len(leaders))
    d.loc[leaders, "volume_ratio"] = np.linspace(2.1, 1.35, len(leaders))
    d.loc[leaders, "up_down_volume_ratio"] = np.linspace(2.6, 1.5, len(leaders))
    d.loc[leaders, "above_50dma"] = 1
    d.loc[leaders, "above_200dma"] = 1
    d.loc[leaders, "catalyst_score"] = np.linspace(90, 72, len(leaders))

    d["relative_3m"] = d["return_3m"] - 2
    d["relative_6m"] = d["return_6m"] - 4
    d["high_52w"] = d["price"] / (1 + d["distance_52w_high_pct"]/100)
    d["pivot_proxy"] = d["price"]/(1+rng.normal(.01,.045,n))
    d["breakout_flag"] = ((d["price"] > d["pivot_proxy"]) & (d["volume_ratio"]>=1.2)).astype(int)
    d["market_cap_rank"] = np.arange(1,n+1)
    d["data_mode"] = "DEMO"
    return d


def _sector_from_company_page(company_url: str, attempts: int = 3):
    """Read Screener's public peer-comparison breadcrumb.

    Screener company pages expose a classification path in the Peer comparison
    section, e.g. Healthcare → Pharmaceuticals & Biotechnology → Pharmaceuticals.
    We use the broadest public bucket as `sector` and the narrowest as `industry`.
    """
    if not isinstance(company_url, str) or not company_url.startswith("http"):
        return "Unclassified", "Unclassified"

    for attempt in range(attempts):
        try:
            r = requests.get(company_url, headers=UA, timeout=20)
            if r.status_code == 429:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")
            peers = soup.find("section", id="peers")
            if not peers:
                return "Unclassified", "Unclassified"

            crumbs = []
            for a in peers.find_all("a", href=True):
                href = a.get("href", "")
                label = re.sub(r"\\s+", " ", a.get_text(" ", strip=True)).strip()
                if href.startswith("/market/") and label:
                    # Ignore peer-table controls and de-duplicate consecutive labels.
                    if not crumbs or crumbs[-1][0] != label:
                        crumbs.append((label, href))

            if not crumbs:
                return "Unclassified", "Unclassified"

            # The broad sector has the shallowest /market/... path. The most
            # specific industry has the deepest path.
            crumbs = sorted(crumbs, key=lambda x: len([p for p in x[1].split("/") if p]))
            sector = crumbs[0][0]
            industry = crumbs[-1][0]
            return sector, industry
        except Exception:
            if attempt < attempts - 1:
                time.sleep(0.6 * (attempt + 1))
    return "Unclassified", "Unclassified"


def _enrich_public_sectors(d: pd.DataFrame, max_workers: int = 6) -> pd.DataFrame:
    """Add sector/industry classifications to a public Screener universe.

    The work is concurrent but deliberately bounded to be gentle on the public
    website. Failed lookups remain Unclassified rather than inventing a sector.
    """
    x = d.copy()
    x["sector"] = "Unclassified"
    x["industry"] = "Unclassified"

    jobs = [(idx, url) for idx, url in zip(x.index, x.get("company_url", pd.Series(index=x.index, dtype=str)))
            if isinstance(url, str) and url.startswith("http")]
    if not jobs:
        return x

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_sector_from_company_page, url): idx for idx, url in jobs}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                sector, industry = fut.result()
            except Exception:
                sector, industry = "Unclassified", "Unclassified"
            x.at[idx, "sector"] = sector or "Unclassified"
            x.at[idx, "industry"] = industry or sector or "Unclassified"
    return x


def fetch_screener_top250(limit=250):
    """Best-effort reader of a public Screener screen plus public sector breadcrumbs.

    Screener has no public API. The screen provides the universe and visible
    fundamentals; each company page's Peer comparison breadcrumb supplies the
    public sector / industry classification.
    """
    rows=[]
    for page in range(1,15):
        url=SCREENER_URL.format(page=page)
        r=requests.get(url,headers=UA,timeout=20)
        r.raise_for_status()
        soup=BeautifulSoup(r.text,"lxml")
        table=soup.select_one("table.data-table") or soup.find("table")
        if not table:
            break
        for tr in table.select("tbody tr"):
            tds=tr.find_all("td")
            if len(tds)<10:
                continue
            a=tds[1].find("a")
            name=a.get_text(" ",strip=True) if a else tds[1].get_text(" ",strip=True)
            href=a.get("href","") if a else ""
            symbol=""
            m=re.search(r"/company/([^/]+)/",href)
            if m:
                symbol=m.group(1).replace("consolidated","").strip("/")
            rows.append({
                "company":name,
                "symbol":symbol,
                "price":_num(tds[2].get_text(" ",strip=True)),
                "pe":_num(tds[3].get_text(" ",strip=True)),
                "market_cap_cr":_num(tds[4].get_text(" ",strip=True)),
                "div_yield":_num(tds[5].get_text(" ",strip=True)),
                "qtr_net_profit_cr":_num(tds[6].get_text(" ",strip=True)),
                "qtr_profit_yoy":_num(tds[7].get_text(" ",strip=True)),
                "sales_qtr_cr":_num(tds[8].get_text(" ",strip=True)),
                "sales_yoy":_num(tds[9].get_text(" ",strip=True)),
                "roce":_num(tds[10].get_text(" ",strip=True)) if len(tds)>10 else np.nan,
                "company_url":"https://www.screener.in"+href if href.startswith("/") else href,
            })
        if len(rows)>=limit:
            break
        time.sleep(.25)
    d=pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("Could not read the public Screener screen.")
    d=d.sort_values("market_cap_cr",ascending=False).drop_duplicates("company").head(limit).reset_index(drop=True)
    d["market_cap_rank"]=np.arange(1,len(d)+1)

    # Correct the previous behaviour that hard-coded every public stock as
    # 'Unclassified'. Classification now comes from Screener's own public
    # Peer-comparison taxonomy on each company page.
    d=_enrich_public_sectors(d, max_workers=6)

    d["eps_yoy"]=d["qtr_profit_yoy"]
    d["roe"]=np.nan
    d["eps_cagr_3y"]=np.nan
    d["sales_cagr_3y"]=np.nan
    d["institutional_holding"]=np.nan
    d["institutional_change"]=np.nan
    d["eps_acceleration"]=np.nan
    d["opm_change"]=np.nan
    d["cfo_positive"]=np.nan
    d["catalyst_score"]=np.nan
    d["data_mode"]="PUBLIC SCREENER + LIVE MARKET"
    return d


def read_screener_csv(file):
    d=pd.read_csv(file)
    aliases={
        "Name":"company","NSE Code":"symbol","Industry":"sector","Market Capitalization":"market_cap_cr",
        "Mar Cap Rs.Cr.":"market_cap_cr","Current Price":"price","CMP Rs.":"price",
        "Qtr Profit Var %":"qtr_profit_yoy","Qtr Sales Var %":"sales_yoy","Return on capital employed":"roce",
        "ROCE %":"roce","Return on equity":"roe","ROE %":"roe","Sales growth 3Years":"sales_cagr_3y",
        "Profit growth 3Years":"eps_cagr_3y","FII holding":"fii_holding","DII holding":"dii_holding"
    }
    d=d.rename(columns={k:v for k,v in aliases.items() if k in d.columns})
    if "company" not in d.columns:
        raise ValueError("CSV needs a Name/company column.")
    if "market_cap_cr" in d.columns:
        d=d.sort_values("market_cap_cr",ascending=False).head(250)
    else:
        d=d.head(250)
    if "sector" not in d.columns:
        d["sector"]="Unclassified"
    if "eps_yoy" not in d.columns and "qtr_profit_yoy" in d.columns:
        d["eps_yoy"]=d["qtr_profit_yoy"]
    if "fii_holding" in d.columns or "dii_holding" in d.columns:
        fii=pd.to_numeric(d.get("fii_holding",0),errors="coerce").fillna(0)
        dii=pd.to_numeric(d.get("dii_holding",0),errors="coerce").fillna(0)
        d["institutional_holding"]=fii+dii
    d["data_mode"]="SCREENER CSV + LIVE MARKET"
    return d.reset_index(drop=True)


def market_metrics(symbols):
    try:
        import yfinance as yf
    except Exception:
        return pd.DataFrame({"symbol":symbols})
    out=[]
    for raw in symbols:
        rec={"symbol":raw}
        if not isinstance(raw,str) or not raw.strip():
            out.append(rec); continue
        ticker=raw.strip().upper()
        if not ticker.endswith(".NS"):
            ticker += ".NS"
        try:
            h=yf.download(ticker,period="1y",interval="1d",auto_adjust=False,progress=False,threads=False)
            if h is None or len(h)<60:
                out.append(rec); continue
            def col(name):
                x=h[name]
                return x.iloc[:,0] if isinstance(x,pd.DataFrame) else x
            c=col("Close").dropna().astype(float); hi=col("High").dropna().astype(float); v=col("Volume").fillna(0).astype(float)
            px=float(c.iloc[-1]); rec["price_live"]=px
            rec["return_3m"]=(px/float(c.iloc[-64])-1)*100 if len(c)>=64 else np.nan
            rec["return_6m"]=(px/float(c.iloc[-127])-1)*100 if len(c)>=127 else np.nan
            rec["return_12m"]=(px/float(c.iloc[0])-1)*100 if len(c)>=200 else np.nan
            high52=float(hi.max()); rec["distance_52w_high_pct"]=(px/high52-1)*100 if high52 else np.nan
            ma50=float(c.tail(50).mean()); ma200=float(c.tail(200).mean()) if len(c)>=200 else np.nan
            rec["above_50dma"]=int(px>ma50); rec["above_200dma"]=int(px>ma200) if pd.notna(ma200) else np.nan
            avg50=float(v.tail(50).mean()); rec["volume_ratio"]=float(v.iloc[-1]/avg50) if avg50>0 else np.nan
            ret=c.pct_change(); up=float(v.where(ret>0,0).tail(65).sum()); down=float(v.where(ret<0,0).tail(65).sum())
            rec["up_down_volume_ratio"]=up/down if down>0 else np.nan
            prior=hi.iloc[-51:-1] if len(hi)>=51 else hi.iloc[:-1]; pivot=float(prior.max()) if len(prior) else np.nan
            rec["pivot_proxy"]=pivot; rec["breakout_flag"]=int(px>pivot and rec["volume_ratio"]>=1.2) if pd.notna(pivot) else np.nan
            out.append(rec)
        except Exception:
            out.append(rec)
    return pd.DataFrame(out)


def merge_live_market(d: pd.DataFrame, max_stocks=250):
    if "symbol" not in d.columns:
        return d
    m=market_metrics(d["symbol"].head(max_stocks).tolist())
    if m.empty:
        return d
    x=d.merge(m,on="symbol",how="left")
    if "price_live" in x.columns:
        x["price"]=x["price_live"].combine_first(pd.to_numeric(x.get("price"),errors="coerce"))
    x["relative_3m"]=pd.to_numeric(x.get("return_3m"),errors="coerce")
    x["relative_6m"]=pd.to_numeric(x.get("return_6m"),errors="coerce")
    return x


def market_regime():
    try:
        import yfinance as yf
        h=yf.download("^NSEI",period="1y",interval="1d",auto_adjust=False,progress=False,threads=False)
        c=h["Close"]; c=c.iloc[:,0] if isinstance(c,pd.DataFrame) else c; c=c.dropna().astype(float)
        px=float(c.iloc[-1]); ma50=float(c.tail(50).mean()); ma200=float(c.tail(200).mean()) if len(c)>=200 else ma50
        score=50+(20 if px>ma50 else -20)+(20 if px>ma200 else -20)
        slope=float(c.tail(50).iloc[-1]/c.tail(50).iloc[0]-1) if len(c)>=50 else 0
        score += 10 if slope>0 else -10; score=float(np.clip(score,0,100))
        label="Confirmed uptrend proxy" if score>=75 else ("Uptrend under pressure" if score>=45 else "Correction / weak market")
        return score,label,px
    except Exception:
        return 50.0,"Market feed unavailable",np.nan


def score(d: pd.DataFrame, market_score=50.0):
    x=d.copy()
    needed=["eps_yoy","sales_yoy","eps_acceleration","opm_change","eps_cagr_3y","sales_cagr_3y","roe","cfo_positive",
            "catalyst_score","volume_ratio","up_down_volume_ratio","breakout_flag","return_12m","relative_6m","relative_3m",
            "distance_52w_high_pct","institutional_holding","institutional_change","above_50dma","above_200dma","pivot_proxy"]
    for c in needed:
        if c not in x.columns: x[c]=np.nan
    # C
    x["_c1"]=percentile(x["eps_yoy"]); x["_c2"]=percentile(x["sales_yoy"]); x["_c3"]=percentile(x["eps_acceleration"]); x["_c4"]=percentile(x["opm_change"])
    x["C"]=weighted_available(x,{"_c1":.45,"_c2":.25,"_c3":.2,"_c4":.1})
    # A, with ROCE fallback when richer annual fields are absent
    x["_a1"]=percentile(x["eps_cagr_3y"]); x["_a2"]=percentile(x["sales_cagr_3y"]); x["_a3"]=percentile(x["roe"])
    x["_a4"]=percentile(x["roce"] if "roce" in x.columns else pd.Series(np.nan,index=x.index))
    x["A"]=weighted_available(x,{"_a1":.35,"_a2":.25,"_a3":.20,"_a4":.20})
    # N
    x["_n1"]=percentile(x["distance_52w_high_pct"]); x["_n2"]=percentile(x["eps_acceleration"]); x["_n3"]=pd.to_numeric(x["catalyst_score"],errors="coerce").clip(0,100)
    x["N"]=weighted_available(x,{"_n1":.45,"_n2":.25,"_n3":.30})
    # S
    x["_s1"]=percentile(x["volume_ratio"]); x["_s2"]=percentile(x["up_down_volume_ratio"]); x["_s3"]=pd.to_numeric(x["breakout_flag"],errors="coerce").map({1:100.0,0:35.0})
    x["S"]=weighted_available(x,{"_s1":.4,"_s2":.4,"_s3":.2})
    # L
    x["_l1"]=percentile(x["return_12m"]); x["_l2"]=percentile(x["relative_6m"]); x["_l3"]=percentile(x["relative_3m"]); x["_l4"]=percentile(x["distance_52w_high_pct"])
    x["L"]=weighted_available(x,{"_l1":.4,"_l2":.25,"_l3":.2,"_l4":.15}); x["rs_proxy"]=x["_l1"].round()
    # I
    x["_i1"]=percentile(x["institutional_holding"]); x["_i2"]=percentile(x["institutional_change"])
    x["I"]=weighted_available(x,{"_i1":.55,"_i2":.45})
    # coverage
    x["factor_coverage"] = x[list("CANSLI")].notna().sum(axis=1)/6*100
    x["equal_quality"]=weighted_available(x,EQUAL_WEIGHTS)
    x["literature_quality"]=weighted_available(x,LITERATURE_WEIGHTS)
    # sector strength
    x["sector"]=x.get("sector","Unclassified").fillna("Unclassified").replace("","Unclassified")
    sec=x.groupby("sector",as_index=False).agg(med6=("return_6m","median"),medrs=("rs_proxy","median"),breadth=("above_50dma","mean"),avg=("literature_quality","mean"),stocks=("company","count"))
    sec["sector_strength"]=(.35*percentile(sec["med6"])+.25*percentile(sec["medrs"])+.15*percentile(sec["breadth"])+.25*percentile(sec["avg"])).clip(0,100)
    sec["sector_rank"]=sec["sector_strength"].rank(ascending=False,method="dense")
    x=x.merge(sec[["sector","sector_strength","sector_rank"]],on="sector",how="left")
    # actionability
    price=pd.to_numeric(x.get("price"),errors="coerce"); pivot=pd.to_numeric(x["pivot_proxy"],errors="coerce"); vr=pd.to_numeric(x["volume_ratio"],errors="coerce")
    dist=(price/pivot-1)*100; x["distance_from_pivot_pct"]=dist
    pivot_score=pd.Series(np.select([(dist>=0)&(dist<=5),(dist<0)&(dist>=-5),(dist>5)&(dist<=10)],[100,80,50],default=25),index=x.index,dtype=float)
    trend=(pd.to_numeric(x["above_50dma"],errors="coerce").fillna(0)*50+pd.to_numeric(x["above_200dma"],errors="coerce").fillna(0)*50)
    volscore=(percentile(vr)*.65+np.where(vr>=1.2,35,10)).clip(0,100)
    x["actionability"]=(.5*pivot_score+.25*trend+.25*volscore).clip(0,100)
    x["setup"]=np.select([(dist>=0)&(dist<=5)&(vr>=1.2),(dist>=0)&(dist<=5),(dist<0)&(dist>=-5),(dist>5)&(dist<=10)],["Breakout / buy-zone proxy","In buy zone","Near pivot","Extended"],default="Not actionable")
    for q in ["equal","literature"]:
        quality=x[f"{q}_quality"]
        quality_m=.90*quality+.10*float(market_score)
        x[f"{q}_final"]=(.70*quality_m+.15*x["sector_strength"]+.15*x["actionability"]).clip(0,100)
    x["rank_literature"]=x["literature_final"].rank(ascending=False,method="min")
    return x


def custom_final(x: pd.DataFrame, weights: Dict[str,float], market_score=50.0):
    w={k:max(float(weights.get(k,0)),0) for k in "CANSLI"}; s=sum(w.values()) or 1; w={k:v/s for k,v in w.items()}
    q=weighted_available(x,w); qm=.90*q+.10*market_score
    return (.70*qm+.15*x["sector_strength"]+.15*x["actionability"]).clip(0,100)


def verdict(row, score_col):
    s=float(row.get(score_col,0) or 0); a=float(row.get("actionability",0) or 0); cov=float(row.get("factor_coverage",0) or 0)
    if cov<55: return "Insufficient data"
    if s>=80 and a>=72: return "Top actionable candidate"
    if s>=75: return "High-quality watchlist"
    if s>=65: return "Research candidate"
    return "Not currently preferred"

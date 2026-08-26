from __future__ import annotations

from datetime import datetime
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import (
    EQUAL_WEIGHTS, LITERATURE_WEIGHTS, custom_final, fetch_screener_top250,
    market_regime, merge_live_market, read_screener_csv, score, synthesize_demo, verdict
)

st.set_page_config(page_title="CANSLIM India", page_icon="◉", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root{--ink:#1d1d1f;--muted:#6e6e73;--blue:#0071e3;--bg:#f5f5f7;--card:#fff;--line:#d2d2d7;--green:#0a7f3f;--amber:#9a6700;--red:#b42318}
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",sans-serif;color:var(--ink)}
.stApp{background:var(--bg)}
.block-container{max-width:1440px;padding-top:1rem;padding-bottom:3rem}
#MainMenu,footer,header{visibility:hidden}
.hero{background:linear-gradient(145deg,#ffffff 0%,#f7f7f9 70%);border:1px solid #e8e8ed;border-radius:32px;padding:48px 52px;margin:8px 0 22px;overflow:hidden}
.eyebrow{font-size:.78rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:700}
.hero h1{font-size:clamp(2.7rem,5vw,5.7rem);line-height:.94;letter-spacing:-.055em;margin:.35rem 0 1rem;font-weight:800}
.hero p{font-size:1.18rem;line-height:1.5;color:var(--muted);max-width:850px}
.pill{display:inline-block;padding:7px 11px;border-radius:999px;background:#e8f2ff;color:#0057b8;font-size:.78rem;font-weight:700;margin-right:6px}
.metric-card{background:#fff;border:1px solid #e5e5ea;border-radius:22px;padding:22px;min-height:124px}
.metric-label{font-size:.78rem;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;font-weight:700}
.metric-value{font-size:2rem;font-weight:800;letter-spacing:-.04em;margin-top:.4rem}
.metric-sub{font-size:.86rem;color:var(--muted);margin-top:.25rem}
.section-title{font-size:2rem;font-weight:800;letter-spacing:-.035em;margin:1.4rem 0 .2rem}
.section-sub{color:var(--muted);margin-bottom:1rem}
.stock-card{background:#fff;border:1px solid #e5e5ea;border-radius:22px;padding:20px;height:100%}
.stock-title{font-size:1.1rem;font-weight:800}.stock-meta{color:var(--muted);font-size:.86rem}.score{font-size:2.4rem;font-weight:850;letter-spacing:-.05em}
.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}
.small-note{font-size:.82rem;color:var(--muted);line-height:1.45}
[data-testid="stDataFrame"]{background:#fff;border-radius:18px;overflow:hidden;border:1px solid #e5e5ea}
div[data-baseweb="select"]>div, .stTextInput input{background:#fff!important;border-radius:14px!important}
.stButton>button,.stDownloadButton>button{border-radius:999px!important;font-weight:700!important;min-height:44px!important}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:#ececf0;border-radius:999px;padding:4px;width:max-content;max-width:100%;overflow:auto}
.stTabs [data-baseweb="tab"]{border-radius:999px;padding:8px 16px;height:auto}
.stTabs [aria-selected="true"]{background:#fff}
hr{border:none;border-top:1px solid #e5e5ea;margin:1.6rem 0}
@media(max-width:700px){.hero{padding:30px 24px;border-radius:24px}.hero p{font-size:1rem}.metric-card{min-height:105px}.block-container{padding-left:.8rem;padding-right:.8rem}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_public_universe_v3():
    return fetch_screener_top250(250)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_live_market(df_serialized: str):
    d=pd.read_json(io.StringIO(df_serialized), orient="split")
    return merge_live_market(d)

@st.cache_data(ttl=900, show_spinner=False)
def cached_regime():
    return market_regime()

if "raw" not in st.session_state:
    st.session_state.raw=synthesize_demo()
    st.session_state.source="Demo data"
    st.session_state.mscore=62.0
    st.session_state.mlabel="Demo — uptrend/neutral proxy"
    st.session_state.nifty=np.nan
if "watch" not in st.session_state: st.session_state.watch=set()

# HERO
st.markdown("""
<div class="hero">
  <div class="eyebrow">CANSLIM INDIA • TOP 250</div>
  <h1>Find leaders.<br>Not noise.</h1>
  <p>An explainable growth-stock research website for India's 250 largest companies by market value. It blends Screener fundamentals, live market behaviour, CAN SLIM scoring, sector strength and entry-quality signals.</p>
  <div style="margin-top:18px"><span class="pill">Explainable scoring</span><span class="pill">Sector winners</span><span class="pill">Live market overlay</span><span class="pill">3 weighting models</span></div>
</div>
""", unsafe_allow_html=True)

# DATA BAR
with st.expander("Data & refresh controls", expanded=st.session_state.source=="Demo data"):
    c1,c2,c3,c4=st.columns([1.2,1.1,1.2,.8])
    with c1:
        mode=st.selectbox("Source",["Demo","Public Screener + live market","Upload Screener CSV + live market"])
    with c2:
        uploaded=st.file_uploader("Screener CSV",type=["csv"],label_visibility="visible") if mode.startswith("Upload") else None
    with c3:
        live_prices=st.checkbox("Refresh live price/volume",value=True,disabled=(mode=="Demo"))
        st.caption("Market data is cached to keep the free-hosted app responsive.")
    with c4:
        st.write("")
        if st.button("Refresh data",type="primary",use_container_width=True):
            try:
                if mode=="Demo":
                    st.session_state.raw=synthesize_demo(); st.session_state.source="Demo data"; st.session_state.mscore=62; st.session_state.mlabel="Demo — uptrend/neutral proxy"; st.session_state.nifty=np.nan
                elif mode.startswith("Public"):
                    with st.spinner("Reading the top-250 Screener universe..."):
                        d=cached_public_universe_v3()
                    if live_prices:
                        with st.spinner("Refreshing market history..."):
                            d=cached_live_market(d.to_json(orient="split"))
                    classified = int(((d.get("sector", pd.Series("Unclassified", index=d.index)) != "Unclassified") & d.get("sector", pd.Series("", index=d.index)).notna()).sum())
                    inst_loaded = int(pd.to_numeric(d.get("institutional_holding", pd.Series(np.nan, index=d.index)), errors="coerce").notna().sum())
                    if classified < max(1, int(len(d)*0.80)) or inst_loaded < max(1, int(len(d)*0.80)):
                        st.warning(f"Public metadata loaded: sectors {classified}/{len(d)} • institutional sponsorship {inst_loaded}/{len(d)}. Screener may rate-limit some company pages; refresh later to fill remaining gaps.")
                    else:
                        st.success(f"Public metadata loaded: sectors {classified}/{len(d)} • institutional sponsorship {inst_loaded}/{len(d)}.")
                    ms,ml,ni=cached_regime(); st.session_state.raw=d; st.session_state.source="Public Screener + live market"; st.session_state.mscore=ms; st.session_state.mlabel=ml; st.session_state.nifty=ni
                else:
                    if uploaded is None: st.error("Upload a Screener export first.")
                    else:
                        d=read_screener_csv(uploaded)
                        if live_prices:
                            with st.spinner("Refreshing market history..."):
                                d=cached_live_market(d.to_json(orient="split"))
                        ms,ml,ni=cached_regime(); st.session_state.raw=d; st.session_state.source="Screener CSV + live market"; st.session_state.mscore=ms; st.session_state.mlabel=ml; st.session_state.nifty=ni
                st.rerun()
            except Exception as e:
                st.error(f"Refresh failed: {e}. The previous dataset was kept.")
    st.caption("Screener has no public API; the public-page connector is best-effort. For a stable hosted project, use a Screener export CSV for fundamentals and refresh market data separately.")

base=score(st.session_state.raw, st.session_state.mscore)

# MODEL
m1,m2,m3=st.columns([1.2,1.2,2])
with m1:
    model=st.selectbox("Ranking model",["Literature","Equal","Custom"],index=0)
with m2:
    min_cov=st.slider("Minimum factor coverage",0,100,50,5)
custom_weights=LITERATURE_WEIGHTS.copy()
if model=="Custom":
    with m3:
        st.caption("Custom weights")
        cc=st.columns(6)
        for i,k in enumerate("CANSLI"):
            custom_weights[k]=cc[i].number_input(k,min_value=0,max_value=40,value=int(LITERATURE_WEIGHTS[k]*100),step=1)/100
    base["selected_score"]=custom_final(base,custom_weights,st.session_state.mscore)
elif model=="Equal": base["selected_score"]=base["equal_final"]
else: base["selected_score"]=base["literature_final"]
base["rank"]=base["selected_score"].rank(ascending=False,method="min")
base["verdict"]=base.apply(lambda r:verdict(r,"selected_score"),axis=1)
ranked=base[base["factor_coverage"]>=min_cov].sort_values("selected_score",ascending=False)

# KPIs
winners=ranked.loc[ranked.groupby("sector")["selected_score"].idxmax()] if len(ranked) else ranked
strong=ranked.groupby("sector")["sector_strength"].max().sort_values(ascending=False) if len(ranked) else pd.Series(dtype=float)
kpis=st.columns(5)
vals=[
    ("Universe",len(base),"Top companies analysed"),
    ("High-quality",int((ranked["selected_score"]>=75).sum()),"Score ≥ 75"),
    ("Actionable",int((ranked["verdict"]=="Top actionable candidate").sum()),"Quality + setup"),
    ("Strongest sector",strong.index[0] if len(strong) else "—",f"Strength {strong.iloc[0]:.0f}" if len(strong) else "—"),
    ("Market regime",f"{st.session_state.mscore:.0f}/100",st.session_state.mlabel),
]
for col,(lab,val,sub) in zip(kpis,vals):
    col.markdown(f'<div class="metric-card"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div><div class="metric-sub">{sub}</div></div>',unsafe_allow_html=True)

st.caption(f"Source: {st.session_state.source} • Updated in this session: {datetime.now().strftime('%d %b %Y, %H:%M')} • Research tool, not personalized advice")

# NAV
pages=st.tabs(["Overview","Sector Scanner","Top 250","Stock Detail","Model Lab","Methodology & Deploy"])

with pages[0]:
    st.markdown('<div class="section-title">Today’s research map</div><div class="section-sub">Start with sector leadership, then drill into the highest-quality setups.</div>',unsafe_allow_html=True)
    a,b=st.columns([1.05,.95])
    with a:
        sec=ranked.groupby("sector",as_index=False).agg(Strength=("sector_strength","max"),Average=("selected_score","mean"),Stocks=("company","count")).sort_values("Strength",ascending=False).head(15)
        fig=px.bar(sec.sort_values("Strength"),x="Strength",y="sector",orientation="h",hover_data=["Average","Stocks"],labels={"sector":""})
        fig.update_traces(marker_color="#1d1d1f"); fig.update_layout(height=470,template="plotly_white",margin=dict(l=10,r=10,t=10,b=10),xaxis_range=[0,100])
        st.plotly_chart(fig,use_container_width=True)
    with b:
        st.markdown("#### Top candidates")
        for _,r in ranked.head(6).iterrows():
            cls="good" if r["verdict"]=="Top actionable candidate" else "warn" if r["selected_score"]>=75 else ""
            st.markdown(f'<div class="stock-card"><div class="stock-meta">#{int(r["rank"])} • {r["sector"]}</div><div class="stock-title">{r["company"]}</div><div class="score {cls}">{r["selected_score"]:.1f}</div><div class="small-note">{r["verdict"]} · {r["setup"]}</div></div>',unsafe_allow_html=True)
    st.markdown("#### Sector winners")
    show=winners.sort_values("selected_score",ascending=False)[["sector","company","selected_score","sector_strength","actionability","factor_coverage","setup","verdict"]].copy()
    st.dataframe(show,hide_index=True,use_container_width=True,column_config={
        "selected_score":st.column_config.ProgressColumn("Final score",min_value=0,max_value=100,format="%.1f"),
        "sector_strength":st.column_config.ProgressColumn("Sector strength",min_value=0,max_value=100,format="%.0f"),
        "actionability":st.column_config.ProgressColumn("Entry quality",min_value=0,max_value=100,format="%.0f"),
        "factor_coverage":st.column_config.ProgressColumn("Data coverage",min_value=0,max_value=100,format="%.0f%%")})

with pages[1]:
    st.markdown('<div class="section-title">Sector Scanner</div><div class="section-sub">Find the strongest candidate inside each sector, then check whether the entry setup is actually usable.</div>',unsafe_allow_html=True)
    sectors=sorted(ranked["sector"].dropna().unique().tolist())
    c1,c2,c3=st.columns([1.3,1,1])
    sector=c1.selectbox("Sector",sectors if sectors else ["Unclassified"])
    minimum=c2.slider("Minimum score",0,100,60,5,key="sector_min")
    actionable_only=c3.checkbox("Near pivot / buy zone only")
    sub=ranked[(ranked["sector"]==sector)&(ranked["selected_score"]>=minimum)].copy()
    if actionable_only: sub=sub[sub["setup"].isin(["Breakout / buy-zone proxy","In buy zone","Near pivot"])]
    if sub.empty: st.info("No stocks match these filters.")
    else:
        r=sub.iloc[0]
        k=st.columns(4); k[0].metric("Sector rank",f"#{int(r['sector_rank'])}" if pd.notna(r['sector_rank']) else "—"); k[1].metric("Sector strength",f"{r['sector_strength']:.0f}"); k[2].metric("Top candidate",r['company']); k[3].metric("Top score",f"{r['selected_score']:.1f}")
        fig=px.bar(sub.head(12).sort_values("selected_score"),x="selected_score",y="company",orientation="h",hover_data=list("CANSLI"),labels={"selected_score":"Score","company":""})
        fig.update_traces(marker_color="#0071e3"); fig.update_layout(template="plotly_white",height=420,xaxis_range=[0,100],margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(sub[["company","symbol","selected_score","C","A","N","S","L","I","actionability","setup","verdict"]],hide_index=True,use_container_width=True)

with pages[2]:
    st.markdown('<div class="section-title">Top 250 leaderboard</div><div class="section-sub">Search, filter and export the complete research universe.</div>',unsafe_allow_html=True)
    f1,f2,f3=st.columns([1.4,1,1])
    q=f1.text_input("Search company or symbol")
    ss=f2.multiselect("Sector",sorted(ranked["sector"].dropna().unique()))
    vv=f3.multiselect("Verdict",sorted(ranked["verdict"].dropna().unique()))
    board=ranked.copy()
    if q:
        qq=q.lower(); board=board[board["company"].astype(str).str.lower().str.contains(qq)|board["symbol"].astype(str).str.lower().str.contains(qq)]
    if ss: board=board[board["sector"].isin(ss)]
    if vv: board=board[board["verdict"].isin(vv)]
    cols=["rank","company","symbol","sector","market_cap_cr","selected_score","C","A","N","S","L","I","sector_strength","actionability","factor_coverage","setup","verdict"]
    st.dataframe(
        board[cols],
        hide_index=True,
        use_container_width=True,
        height=650,
        column_config={
            "selected_score": st.column_config.ProgressColumn(
                "Final score", min_value=0, max_value=100, format="%.1f"
            ),
            "factor_coverage": st.column_config.ProgressColumn(
                "Coverage", min_value=0, max_value=100, format="%.0f%%"
            ),
        },
    )
    st.download_button("Download current ranking",board.to_csv(index=False).encode(),"canslim_india_ranking.csv","text/csv")

with pages[3]:
    st.markdown('<div class="section-title">Stock Detail</div><div class="section-sub">See exactly why a stock ranks where it does.</div>',unsafe_allow_html=True)
    opts=ranked["company"].tolist() if len(ranked) else base["company"].tolist()
    company=st.selectbox("Company",opts)
    r=base[base["company"]==company].iloc[0]
    k=st.columns(5); k[0].metric("Final score",f"{r['selected_score']:.1f}"); k[1].metric("Rank",f"#{int(r['rank'])}" if pd.notna(r['rank']) else "—"); k[2].metric("Sector",r['sector']); k[3].metric("Actionability",f"{r['actionability']:.0f}"); k[4].metric("Coverage",f"{r['factor_coverage']:.0f}%")
    lft,rgt=st.columns([.9,1.1])
    with lft:
        vals=[float(r.get(k,0) if pd.notna(r.get(k,np.nan)) else 0) for k in "CANSLI"]
        fig=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=list("CANSLI")+["C"],fill="toself",line_color="#0071e3",fillcolor="rgba(0,113,227,.18)"))
        fig.update_layout(template="plotly_white",polar=dict(radialaxis=dict(visible=True,range=[0,100])),showlegend=False,height=420,margin=dict(l=30,r=30,t=20,b=20))
        st.plotly_chart(fig,use_container_width=True)
    with rgt:
        strengths={k:r.get(k,np.nan) for k in "CANSLI" if pd.notna(r.get(k,np.nan))}; top=sorted(strengths,key=strengths.get,reverse=True)[:2]; weak=sorted(strengths,key=strengths.get)[:1]
        names={"C":"current earnings","A":"annual earnings","N":"new/catalyst proxy","S":"supply-demand","L":"price leadership","I":"institutional sponsorship"}
        why=(f"{r['company']} scores {r['selected_score']:.1f}/100. " + ("Its strongest areas are "+" and ".join(names[x] for x in top)+". " if top else "") + (f"The weakest available area is {names[weak[0]]}. " if weak else "") + f"The current setup is '{r['setup']}'.")
        st.markdown(f'<div class="stock-card"><div class="stock-meta">{r["sector"]} • {r.get("symbol","")}</div><div class="stock-title">{r["company"]}</div><div class="score">{r["selected_score"]:.1f}</div><div class="small-note">{why}</div><hr><b>{r["verdict"]}</b><br><span class="small-note">Research candidate only — not personalized investment advice.</span></div>',unsafe_allow_html=True)
        e1,e2,e3=st.columns(3); e1.metric("Pivot distance",f"{r['distance_from_pivot_pct']:.1f}%" if pd.notna(r['distance_from_pivot_pct']) else "—"); e2.metric("Volume ratio",f"{r.get('volume_ratio',np.nan):.2f}x" if pd.notna(r.get('volume_ratio',np.nan)) else "—"); e3.metric("RS proxy",f"{r.get('rs_proxy',np.nan):.0f}" if pd.notna(r.get('rs_proxy',np.nan)) else "—")
        if st.button("Add / remove watchlist"):
            key=str(r.get("symbol") or r["company"])
            if key in st.session_state.watch: st.session_state.watch.remove(key)
            else: st.session_state.watch.add(key)
            st.rerun()
    evidence=pd.DataFrame({"Metric":["Quarter profit/EPS YoY","Quarter sales YoY","3Y EPS growth","3Y sales growth","ROE","ROCE","FII holding","DII holding","Institutional holding (FII+DII)","Institutional change QoQ","12M return","6M return","Distance from 52W high","Volume / 50D avg"],"Value":[r.get("eps_yoy"),r.get("sales_yoy"),r.get("eps_cagr_3y"),r.get("sales_cagr_3y"),r.get("roe"),r.get("roce"),r.get("fii_holding"),r.get("dii_holding"),r.get("institutional_holding"),r.get("institutional_change"),r.get("return_12m"),r.get("return_6m"),r.get("distance_52w_high_pct"),r.get("volume_ratio")]})
    st.dataframe(evidence,hide_index=True,use_container_width=True)

with pages[4]:
    st.markdown('<div class="section-title">Model Lab</div><div class="section-sub">Compare equal weighting, literature weighting and a data-derived model without hiding the assumptions.</div>',unsafe_allow_html=True)
    weights=pd.DataFrame({"Factor":list("CANSLI"),"Equal":[EQUAL_WEIGHTS[k]*100 for k in "CANSLI"],"Literature":[LITERATURE_WEIGHTS[k]*100 for k in "CANSLI"]})
    fig=px.bar(weights.melt("Factor",var_name="Model",value_name="Weight"),x="Factor",y="Weight",color="Model",barmode="group",color_discrete_sequence=["#1d1d1f","#0071e3"])
    fig.update_layout(template="plotly_white",height=360,margin=dict(l=10,r=10,t=10,b=10),yaxis_title="Weight (%)")
    st.plotly_chart(fig,use_container_width=True)
    comp=base[["company","equal_final","literature_final"]].dropna(); comp["re"]=comp["equal_final"].rank(ascending=False); comp["rl"]=comp["literature_final"].rank(ascending=False)
    corr=comp[["re","rl"]].corr().iloc[0,1] if len(comp)>2 else np.nan
    c1,c2=st.columns(2); c1.metric("Rank correlation",f"{corr:.3f}" if pd.notna(corr) else "—"); c2.metric("Top-20 overlap",f"{len(set(comp.nsmallest(20,'re').company)&set(comp.nsmallest(20,'rl').company))}/20")
    st.markdown("#### Data-derived model")
    st.caption("Upload point-in-time historical observations. The app will not fake a data-driven model using today's data and future information.")
    train=st.file_uploader("Historical training CSV",type=["csv"],key="train")
    if train is not None:
        try:
            from sklearn.linear_model import LogisticRegression
            t=pd.read_csv(train); need=list("CANSLI")+["forward_6m_excess_return"]; miss=[c for c in need if c not in t.columns]
            if miss: raise ValueError("Missing: "+", ".join(miss))
            tt=t[need].dropna(); X=tt[list("CANSLI")]; y=(tt["forward_6m_excess_return"]>0).astype(int); mdl=LogisticRegression(max_iter=2000).fit(X,y)
            co=pd.DataFrame({"Factor":list("CANSLI"),"Coefficient":mdl.coef_[0]}); st.dataframe(co,hide_index=True,use_container_width=True)
            prob=mdl.predict_proba(base[list("CANSLI")].fillna(50))[:,1]*100; out=base[["company","sector"]].copy(); out["Probability of 6M excess return > 0"]=prob
            st.dataframe(out.sort_values(out.columns[-1],ascending=False).head(30),hide_index=True,use_container_width=True)
        except Exception as e: st.error(str(e))
    else:
        st.info("Required columns: C, A, N, S, L, I, forward_6m_excess_return.")

with pages[5]:
    st.markdown('<div class="section-title">Methodology & free deployment</div><div class="section-sub">Transparent assumptions make the project defensible.</div>',unsafe_allow_html=True)
    st.markdown("""
### Architecture
**Top-250 Screener universe → fundamentals → live price/volume history → CAN SLIM factor engine → equal/literature/data-derived models → sector strength → actionability → explainable ranking.**

### CAN SLIM proxies
- **C:** current profit/EPS growth, sales growth, acceleration and margin change when available.
- **A:** multi-year earnings/sales growth, ROE and ROCE when available.
- **N:** proximity to new highs + earnings acceleration + optional catalyst score. This is explicitly a proxy, not MarketSmith's proprietary implementation.
- **S:** current-vs-50-day volume, 13-week up/down volume balance and breakout-volume evidence.
- **L:** 12-, 6- and 3-month relative price leadership and proximity to 52-week highs.
- **I:** institutional ownership and ownership change when available.
- **M:** NIFTY market-regime overlay based on 50/200-day trend conditions.

### Quality vs entry
A great company is not automatically a good entry. The app keeps **CAN SLIM quality**, **sector strength** and **actionability** separate, then combines them transparently.

### Free hosting
1. Create a GitHub repository and upload the project folder.
2. Go to **Streamlit Community Cloud** and choose **Create app**.
3. Select your repository, branch and `app.py`.
4. Deploy. The free deployment receives a `streamlit.app` URL and updates when the connected GitHub branch changes.

For a stable public project, keep fundamental data in a Screener export committed to the repository or uploaded by the user, and let the app refresh price/volume data on a cache interval. Public Screener page reading is best-effort rather than an API.
""")
    st.warning("MarketSmith India's proprietary Master Score, exact EPS/RS calculations, pattern-recognition engine and Group Rank formula are not reproduced. This project uses transparent, literature-inspired proxies for academic research.")
    st.error("This website ranks research candidates. It is not a personalized recommendation engine and does not guarantee investment outcomes.")

if st.session_state.watch:
    st.markdown("---")
    st.markdown("#### Watchlist")
    st.write(" • ".join(sorted(st.session_state.watch)))

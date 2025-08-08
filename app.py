"""
Streamlit Portfolio / Projection Dashboard (MVP)

What this app does
- Replaces your two Google Sheet tabs: "Live" (current portfolio) and "Projection" (hard-coded schedule).
- Admin (you) can edit inputs (formerly the blue cells), FX overrides, account tags, and projection table.
- Viewer (your wife) can see everything but cannot edit.
- Data is persisted in Supabase (managed Postgres). Auth is via simple app-level password (admin vs viewer).

How to run locally (5‑minute guide)
1) `pip install -r requirements.txt`
2) Create a file `.streamlit/secrets.toml` (see the SECRETS section below) and paste your values.
3) `streamlit run app.py`

How to deploy online (Streamlit Cloud, 10‑minute guide)
1) Create a free Supabase project at https://supabase.com > New project.
   - In your project, open SQL Editor and run the DDL from the comment block named "SUPABASE SCHEMA" below.
   - In Settings > API, copy your Project URL and Service Role key.
2) Create a new public GitHub repo. Add 2 files: `app.py` (this file) and `requirements.txt` (see bottom of file).
3) Go to https://share.streamlit.io (or Streamlit Cloud). Deploy your GitHub repo.
4) In your deployed app > Settings > Secrets, paste the same content you would put into `.streamlit/secrets.toml` (see SECRETS below).
5) Open the app. Log in with Admin password. Upload your Excel (the one you shared) on the Projection page to import the schedule once, and on the Live page to import account rows if you prefer.
6) Share the link + Viewer password with your wife.

SUPABASE SCHEMA — paste this in Supabase SQL Editor and RUN once
-----------------------------------------------------------------
-- Enable pgcrypto for UUIDs, if not already enabled
create extension if not exists pgcrypto;

create table if not exists accounts (
  id uuid primary key default gen_random_uuid(),
  created_at timestamp with time zone default now(),
  type text not null,                  -- e.g., Cash, Equity, Bitcoin
  institution text not null,           -- e.g., HSBC, IBKR, Kraken
  currency text not null,              -- e.g., CHF, USD, GBP, BTC
  value_lc numeric not null,           -- Value in local currency (your input)
  class_tag text not null check (class_tag in ('Global Equity','Swiss Equity','Cash+Bonds')),
  is_liquid boolean not null default false,
  notes text
);

create table if not exists settings (
  key text primary key,
  value jsonb
);

create table if not exists fx_rates (
  code text primary key,               -- currency code e.g. USD, EUR, BTC
  rate_to_chf numeric not null,        -- base live rate to CHF
  override_rate numeric,               -- optional admin override
  source text,
  updated_at timestamp with time zone default now()
);

create table if not exists projection_rows (
  dt date primary key,
  cash numeric,
  bitcoin numeric,
  pillar3a numeric,
  pillar2 numeric,
  ibkr numeric,
  pillar1e numeric,
  grand_total numeric
);

-- Seed default settings
insert into settings(key, value) values
  ('assumptions', '{"cash":0.0, "bitcoin":0.08, "pillar3a":0.065, "pillar2":0.0075, "ibkr":0.07, "pillar1e":0.04}')
  on conflict (key) do nothing;

-----------------------------------------------------------------

SECRETS — put this in `.streamlit/secrets.toml` (local) or Streamlit Cloud > Secrets
----------------------------------------------------------------------------------
[supabase]
url = "https://YOUR-PROJECT.supabase.co"
service_role_key = "YOUR_SERVICE_ROLE_KEY"   # keep secret! only on server, never in client-side code

[auth]
admin_password = "SET_A_STRONG_ADMIN_PASSWORD"
viewer_password = "SET_A_VIEWER_PASSWORD_FOR_WIFE"

[api]
fx_base = "CHF"
fx_url = "https://api.exchangerate.host/latest?base=CHF"   # can change base via secrets if needed
btc_url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=chf"

[ui]
brand = "Soumya Portfolio"

"""

import io
import json
import time
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit.runtime.state import SessionState

# Supabase client
try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# -----------------------
# Helpers & Configuration
# -----------------------
APP_BRAND = st.secrets.get("ui", {}).get("brand", "Portfolio App")

def make_client() -> Optional['Client']:
    if create_client is None:
        st.error("supabase-py not installed. Add it to requirements.txt")
        return None
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["service_role_key"]
    return create_client(url, key)

sb = make_client()

# -----------------------
# Auth (simple roles)
# -----------------------
ROLE_ADMIN = "admin"
ROLE_VIEWER = "viewer"

def login_panel():
    st.sidebar.markdown(f"### {APP_BRAND}")
    role = st.session_state.get("role")
    if role in (ROLE_ADMIN, ROLE_VIEWER):
        with st.sidebar.expander("Session"):
            st.write(f"Logged in as **{role}**")
            if st.button("Log out"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.experimental_rerun()
        return role

    st.sidebar.info("Enter the app password to continue.")
    pwd = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if pwd == st.secrets["auth"]["admin_password"]:
            st.session_state["role"] = ROLE_ADMIN
            st.experimental_rerun()
        elif pwd == st.secrets["auth"]["viewer_password"]:
            st.session_state["role"] = ROLE_VIEWER
            st.experimental_rerun()
        else:
            st.sidebar.error("Wrong password.")
    st.stop()

# -----------------------
# Data access layer
# -----------------------

def get_accounts_df() -> pd.DataFrame:
    res = sb.table("accounts").select("*").execute()
    rows = res.data or []
    if not rows:
        return pd.DataFrame(columns=["id","created_at","type","institution","currency","value_lc","class_tag","is_liquid","notes"])  # empty
    df = pd.DataFrame(rows)
    # Ensure column order
    cols = ["id","created_at","type","institution","currency","value_lc","class_tag","is_liquid","notes"]
    return df[cols]

def upsert_accounts(df: pd.DataFrame):
    # Upsert each row by id (if id empty, let DB create id via insert)
    for _, r in df.iterrows():
        payload = {
            "id": r.get("id") or None,
            "type": r["type"],
            "institution": r["institution"],
            "currency": r["currency"],
            "value_lc": float(r["value_lc"]) if pd.notnull(r["value_lc"]) else 0.0,
            "class_tag": r["class_tag"],
            "is_liquid": bool(r["is_liquid"]) if pd.notnull(r["is_liquid"]) else False,
            "notes": r.get("notes"),
        }
        if payload["id"] is None:
            sb.table("accounts").insert(payload).execute()
        else:
            sb.table("accounts").upsert(payload, on_conflict="id").execute()


def delete_account(row_id: str):
    sb.table("accounts").delete().eq("id", row_id).execute()


def get_settings() -> Dict:
    res = sb.table("settings").select("*").execute()
    d = {r["key"]: r["value"] for r in (res.data or [])}
    return d


def update_settings(key: str, value: Dict):
    payload = {"key": key, "value": value}
    sb.table("settings").upsert(payload, on_conflict="key").execute()


def get_fx_table() -> pd.DataFrame:
    res = sb.table("fx_rates").select("*").execute()
    return pd.DataFrame(res.data or [])


def upsert_fx_row(code: str, rate: float, source: str, override: Optional[float] = None):
    payload = {
        "code": code,
        "rate_to_chf": float(rate),
        "override_rate": float(override) if override is not None else None,
        "source": source,
        "updated_at": dt.datetime.utcnow().isoformat(),
    }
    sb.table("fx_rates").upsert(payload, on_conflict="code").execute()


def get_projection_df() -> pd.DataFrame:
    res = sb.table("projection_rows").select("*").order("dt").execute()
    rows = res.data or []
    if not rows:
        cols = ["dt","cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e","grand_total"]
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    return df


def upsert_projection_df(df: pd.DataFrame):
    # Compute grand_total if not provided
    if "grand_total" in df.columns:
        pass
    else:
        df["grand_total"] = df[["cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e"]].sum(axis=1)

    # Upsert rows one by one (simple & safe)
    for _, r in df.iterrows():
        payload = {
            "dt": str(r["dt"]),
            "cash": float(r.get("cash") or 0),
            "bitcoin": float(r.get("bitcoin") or 0),
            "pillar3a": float(r.get("pillar3a") or 0),
            "pillar2": float(r.get("pillar2") or 0),
            "ibkr": float(r.get("ibkr") or 0),
            "pillar1e": float(r.get("pillar1e") or 0),
            "grand_total": float(r.get("grand_total") or 0),
        }
        sb.table("projection_rows").upsert(payload, on_conflict="dt").execute()

# -----------------------
# Live page logic
# -----------------------
CLASS_CHOICES = ["Global Equity", "Swiss Equity", "Cash+Bonds"]

BRIGHT = px.colors.qualitative.Bold + px.colors.qualitative.Set3 + px.colors.qualitative.Vivid


def fetch_live_fx(target_codes: List[str]) -> Dict[str, float]:
    """Fetch live FX to CHF for given codes. BTC handled separately."""
    base = st.secrets["api"].get("fx_base", "CHF").upper()
    fx_url = st.secrets["api"]["fx_url"]
    rates = {base: 1.0}
    try:
        resp = requests.get(fx_url, timeout=10)
        data = resp.json()
        # exchangerate.host returns rates for many currencies vs base
        raw = data.get("rates", {})
        for code in target_codes:
            code = code.upper()
            if code == base:
                rates[code] = 1.0
            elif code in raw:
                # We need code->CHF. If base=CHF, rate is direct. If base != CHF, we'd need to invert; keep base=CHF by default.
                rates[code] = 1.0 / raw[code] if base != "CHF" else raw[code]
        return rates
    except Exception as e:
        st.warning(f"FX fetch failed: {e}")
        return rates


def fetch_btc_chf() -> Optional[float]:
    try:
        resp = requests.get(st.secrets["api"]["btc_url"], timeout=10)
        j = resp.json()
        return float(j.get("bitcoin", {}).get("chf"))
    except Exception as e:
        st.warning(f"BTC price fetch failed: {e}")
        return None


def compute_live_view(accts: pd.DataFrame, fx_tbl: pd.DataFrame, use_live: bool, btc_override: Optional[float]) -> pd.DataFrame:
    df = accts.copy()
    df["currency"] = df["currency"].str.upper()

    # Build rate map: start with DB table
    rate_map = {r["code"].upper(): float(r["override_rate"] or r["rate_to_chf"]) for _, r in fx_tbl.iterrows()}

    # Collect missing currencies and fetch live
    missing = sorted(set(df["currency"]) - set(rate_map.keys()))
    if use_live and missing:
        live = fetch_live_fx(missing)
        for k, v in live.items():
            if k not in rate_map:
                rate_map[k] = v
                upsert_fx_row(k, v, source="live")

    # BTC special-case
    if use_live:
        live_btc = fetch_btc_chf()
        if live_btc:
            upsert_fx_row("BTC", live_btc, source="coingecko")
            rate_map.setdefault("BTC", live_btc)
    if btc_override is not None:
        rate_map["BTC"] = btc_override
        upsert_fx_row("BTC", rate_map.get("BTC", 0), source="override", override=btc_override)

    # Default CHF if currency unknown
    df["rate_to_chf"] = df["currency"].map(rate_map).fillna(1.0)

    # If currency is CHF, rate=1. If BTC, value_lc is in BTC units and we convert by price.
    df["value_chf"] = df.apply(lambda r: float(r["value_lc"]) * float(r["rate_to_chf"]) , axis=1)

    # Rollups
    total = df["value_chf"].sum()
    if total == 0:
        total = 1
    df["pct"] = df["value_chf"] / total

    return df


def live_page(role: str):
    st.title("Live Portfolio")

    use_live = st.toggle("Use live FX/BTC where available", value=True, help="Overrides can still be applied below.")
    btc_override = st.number_input("BTC → CHF override (optional)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    btc_override = btc_override if btc_override > 0 else None

    # Accounts editor
    accts = get_accounts_df()

    if role == ROLE_ADMIN:
        st.subheader("Accounts (edit your blue cells here)")
        if accts.empty:
            st.info("No accounts yet. You can paste rows or import from your Excel (Live tab) below.")

        edited = st.data_editor(
            accts if not accts.empty else pd.DataFrame(
                columns=["id","type","institution","currency","value_lc","class_tag","is_liquid","notes"]
            ),
            num_rows="dynamic",
            column_config={
                "type": st.column_config.SelectboxColumn("Type", options=["Cash","Equity","Bitcoin","Pension"], required=True),
                "class_tag": st.column_config.SelectboxColumn("Class", options=CLASS_CHOICES, required=True),
                "currency": st.column_config.TextColumn("Currency (e.g., CHF, USD, GBP, BTC)", required=True),
                "value_lc": st.column_config.NumberColumn("Value (LC)", required=True, format="%.2f"),
                "is_liquid": st.column_config.CheckboxColumn("Liquid?"),
                "notes": st.column_config.TextColumn("Notes"),
                "id": None,
                "created_at": None,
            },
            hide_index=True,
            use_container_width=True,
            key="accts_editor",
        )
        col_s, col_d = st.columns([1,1])
        with col_s:
            if st.button("Save accounts"):
                upsert_accounts(edited.drop(columns=[c for c in ["created_at"] if c in edited.columns]))
                st.success("Saved.")
                time.sleep(0.6)
                st.experimental_rerun()
        with col_d:
            if not accts.empty:
                to_delete = st.selectbox("Delete row by id (careful)", options=["-"] + accts["id"].astype(str).tolist())
                if to_delete != "-" and st.button("Confirm delete"):
                    delete_account(to_delete)
                    st.success("Deleted.")
                    time.sleep(0.5)
                    st.experimental_rerun()

        st.divider()
        st.subheader("Import accounts from Excel (Live tab)")
        up = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False)
        if up:
            try:
                xl = pd.ExcelFile(up)
                if "Live" not in xl.sheet_names:
                    st.error("Couldn't find a 'Live' sheet in this workbook.")
                else:
                    df_live = xl.parse("Live", header=None)
                    # Heuristic: look for rows that contain account structure like [Type, Name, Value(LC), Value(CHF)...]
                    # We'll try to capture rows where column 1/2 look textual and column 3 numeric
                    rows = []
                    for _, row in df_live.iterrows():
                        try:
                            t, inst, v = str(row[0]).strip(), str(row[1]).strip(), row[2]
                            if t.lower() in ["cash","equity","bitcoin","pension"] and pd.to_numeric(v, errors="coerce") is not None:
                                rows.append({
                                    "type": t.title(),
                                    "institution": inst,
                                    "currency": "CHF",  # default, you can change after import
                                    "value_lc": float(v),
                                    "class_tag": "Cash+Bonds" if t.lower()=="cash" else ("Global Equity" if t.lower() in ["equity","bitcoin"] else "Swiss Equity"),
                                    "is_liquid": t.lower() in ["cash","equity","bitcoin"],
                                })
                        except Exception:
                            pass
                    if rows:
                        st.dataframe(pd.DataFrame(rows))
                        if st.button("Insert imported rows"):
                            upsert_accounts(pd.DataFrame(rows))
                            st.success("Imported.")
                            st.experimental_rerun()
                    else:
                        st.warning("Importer didn't recognize rows. You can paste manually into the editor above.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    # Rates table + calculation
    fx_tbl = get_fx_table()
    live_df = compute_live_view(get_accounts_df(), fx_tbl, use_live=use_live, btc_override=btc_override)

    st.subheader("Portfolio breakdown (CHF)")
    if live_df.empty:
        st.info("Add accounts to see charts.")
        return

    # Charts
    by_inst = live_df.groupby("institution", as_index=False)["value_chf"].sum().sort_values("value_chf", ascending=False)
    fig_bar = px.bar(by_inst, x="institution", y="value_chf", title="By Institution", color_discrete_sequence=BRIGHT)
    st.plotly_chart(fig_bar, use_container_width=True)

    by_class = live_df.groupby("class_tag", as_index=False)["value_chf"].sum()
    fig_pie = px.pie(by_class, names="class_tag", values="value_chf", hole=0.55, title="Allocation by Class", color_discrete_sequence=BRIGHT)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Totals")
    total = live_df["value_chf"].sum()
    liquid = live_df[live_df["is_liquid"]]["value_chf"].sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Grand Total (CHF)", f"{total:,.2f}")
    c2.metric("Liquid Assets (CHF)", f"{liquid:,.2f}")
    c3.metric("# Accounts", f"{len(live_df):,}")

    st.subheader("Detail table")
    show = live_df[["type","institution","currency","value_lc","rate_to_chf","value_chf","class_tag","is_liquid"]].copy()
    show = show.sort_values("value_chf", ascending=False)
    st.dataframe(show, use_container_width=True)

    st.caption("Tip: Change class tags or Liquid? in the editor (admin) to adjust the rollups above.")


# -----------------------
# Projection page
# -----------------------

def projection_page(role: str):
    st.title("Projections (hard-coded schedule)")
    st.write("This mirrors your left-side table. No partial year. You can import once from Excel or edit inline.")

    df = get_projection_df()

    if role == ROLE_ADMIN:
        with st.expander("Import from Excel › 'Projection' sheet"):
            up = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"], key="proj_up")
            if up:
                try:
                    x = pd.ExcelFile(up)
                    if "Projection" not in x.sheet_names:
                        st.error("Couldn't find a 'Projection' sheet.")
                    else:
                        raw = x.parse("Projection", header=None)
                        # Expect headers in row 2, data from row 3 onward similar to your sheet.
                        # We'll scan for a column that looks like dates, then the next 6 numeric columns, then compute total.
                        # Heuristic is intentionally forgiving.
                        candidates = []
                        for r in range(0, min(10, len(raw))):
                            row = raw.iloc[r].tolist()
                            if any(isinstance(v, (dt.date, dt.datetime, pd.Timestamp)) for v in row):
                                candidates.append(r)
                        start_row = candidates[0] if candidates else 2
                        data = raw.iloc[start_row:]
                        # Find first 7 non-null columns
                        data = data.dropna(how='all', axis=1)
                        data.columns = range(data.shape[1])
                        # Expect: [date, cash, bitcoin, pillar3a, pillar2, ibkr, pillar1e, (maybe total)]
                        cols_map = {0:"dt",1:"cash",2:"bitcoin",3:"pillar3a",4:"pillar2",5:"ibkr",6:"pillar1e"}
                        out = data[list(cols_map.keys())].rename(columns=cols_map)
                        out = out[out["dt"].notnull()]
                        out["dt"] = pd.to_datetime(out["dt"]).dt.date
                        out[["cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e"]] = out[["cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e"]].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                        out["grand_total"] = out[["cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e"]].sum(axis=1)
                        st.dataframe(out.head())
                        if st.button("Replace projection table with imported data"):
                            # Clear and upsert (simple way: delete all first)
                            sb.table("projection_rows").delete().neq("dt", None).execute()
                            upsert_projection_df(out)
                            st.success("Projection imported.")
                            st.experimental_rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    st.subheader("Schedule")
    editable = df.copy()
    if role == ROLE_VIEWER:
        st.dataframe(editable, use_container_width=True)
    else:
        edited = st.data_editor(
            editable,
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "dt": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
            },
            key="proj_editor",
        )
        if st.button("Save schedule"):
            upsert_projection_df(edited)
            st.success("Saved.")
            time.sleep(0.5)
            st.experimental_rerun()

    if not df.empty:
        st.subheader("Chart")
        long = df.melt(id_vars=["dt"], value_vars=["cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e"], var_name="bucket", value_name="value")
        fig = px.area(long, x="dt", y="value", color="bucket", title="Projection by bucket", color_discrete_sequence=BRIGHT)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Grand Total over time")
        fig2 = px.line(df, x="dt", y="grand_total", title="Grand Total (CHF)", color_discrete_sequence=BRIGHT)
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button("Download projections CSV", df.to_csv(index=False).encode("utf-8"), file_name="projections.csv")


# -----------------------
# App entry
# -----------------------
role = login_panel()

page = st.sidebar.radio("Navigate", ["Live","Projections"], index=0)
if page == "Live":
    live_page(role)
else:
    projection_page(role)

"""
requirements.txt (create this file next to app.py)
-----------------------------------------------
streamlit==1.36.0
pandas
plotly
requests
supabase
openpyxl
"""

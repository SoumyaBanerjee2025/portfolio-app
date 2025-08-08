# app.py — Streamlit Portfolio / Projection Dashboard (MVP, FX-fixed, robust saves)

import time
import datetime as dt
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st
import plotly.express as px

# --- Supabase client ---
try:
    from supabase import create_client
except Exception:
    create_client = None

# -----------------------
# Config
# -----------------------
APP_BRAND = st.secrets.get("ui", {}).get("brand", "Portfolio App")
ROLE_ADMIN = "admin"
ROLE_VIEWER = "viewer"

CLASS_CHOICES = ["Global Equity", "Swiss Equity", "Cash+Bonds"]
BRIGHT = px.colors.qualitative.Bold + px.colors.qualitative.Set3 + px.colors.qualitative.Vivid

# -----------------------
# Supabase connection
# -----------------------
def make_client():
    if create_client is None:
        st.error("supabase-py not installed. Ensure `requirements.txt` includes `supabase`.")
        return None
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["service_role_key"]
        return create_client(url, key)
    except KeyError:
        st.stop()  # Secrets not set (Cloud shows a nicer message earlier)


sb = make_client()

# -----------------------
# Auth (simple passwords)
# -----------------------
def login_panel() -> str:
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
    res = sb.table("accounts").select("*").order("created_at").execute()
    rows = res.data or []
    cols = ["id","created_at","type","institution","currency","value_lc","class_tag","is_liquid","notes"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols]


def upsert_accounts(df: pd.DataFrame):
    """Sanitize rows, skip blanks, INSERT new rows, UPDATE existing rows."""
    if df is None or df.empty:
        return

    df = df.copy()
    required = ["type", "institution", "currency", "value_lc", "class_tag"]
    for col in required + ["id", "is_liquid", "notes"]:
        if col not in df.columns:
            df[col] = None

    # Drop completely empty editor rows
    df = df[~df[required].isnull().all(axis=1)]

    for _, r in df.iterrows():
        # Skip rows still missing core fields
        if all(pd.isna(r.get(k)) or str(r.get(k)).strip() == "" for k in required):
            continue

        # Clean values
        class_tag = (str(r.get("class_tag", "")).strip()
                     .replace("Cash+Bond", "Cash+Bonds")
                     .replace("Cash + Bonds", "Cash+Bonds"))
        if class_tag not in ("Global Equity", "Swiss Equity", "Cash+Bonds"):
            class_tag = "Cash+Bonds"

        try:
            value_lc = float(r.get("value_lc")) if pd.notnull(r.get("value_lc")) else 0.0
        except Exception:
            value_lc = 0.0

        notes = r.get("notes")
        if notes is None or (isinstance(notes, float) and pd.isna(notes)):
            notes = None
        else:
            notes = str(notes)

        payload = {
            "type": str(r.get("type", "")).strip() or "Cash",
            "institution": str(r.get("institution", "")).strip() or "Unknown",
            "currency": str(r.get("currency", "CHF")).strip().upper() or "CHF",
            "value_lc": value_lc,
            "class_tag": class_tag,
            "is_liquid": bool(r.get("is_liquid")) if pd.notnull(r.get("is_liquid")) else False,
            "notes": notes,
        }

        row_id = r.get("id")
        if row_id and str(row_id).strip():
            # UPDATE existing row
            sb.table("accounts").update(payload).eq("id", str(row_id)).execute()
        else:
            # INSERT new row (let DB auto-generate id)
            sb.table("accounts").insert(payload).execute()


def delete_account(row_id: str):
    sb.table("accounts").delete().eq("id", row_id).execute()


def get_fx_table() -> pd.DataFrame:
    res = sb.table("fx_rates").select("*").execute()
    return pd.DataFrame(res.data or [])


def upsert_fx_row(code: str, rate: float, source: str, override: Optional[float] = None):
    payload = {
        "code": code.upper(),
        "rate_to_chf": float(rate),
        "override_rate": float(override) if override is not None else None,
        "source": source,
        "updated_at": dt.datetime.utcnow().isoformat(),
    }
    sb.table("fx_rates").upsert(payload, on_conflict="code").execute()


def get_projection_df() -> pd.DataFrame:
    res = sb.table("projection_rows").select("*").order("dt").execute()
    rows = res.data or []
    cols = ["dt","cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e","grand_total"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"]).dt.date
    return df


def upsert_projection_df(df: pd.DataFrame):
    if df is None or df.empty:
        return
    df = df.copy()

    # Clean/validate dates
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce").dt.date
    df = df[df["dt"].notnull()]

    # Clean numeric columns (strip commas/spaces, coerce to numbers)
    num_cols = ["cash", "bitcoin", "pillar3a", "pillar2", "ibkr", "pillar1e", "grand_total"]
    for c in num_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Always recompute grand_total server-side
    df["grand_total"] = df[["cash", "bitcoin", "pillar3a", "pillar2", "ibkr", "pillar1e"]].sum(axis=1)

    for _, r in df.iterrows():
        payload = {
            "dt": str(pd.to_datetime(r["dt"]).date()),
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
# Live page helpers
# -----------------------
def fetch_live_fx(target_codes: List[str]) -> Dict[str, float]:
    """
    Return 1 <code> -> CHF. Try Frankfurter (ECB) first, fall back to exchangerate.host.
    """
    rates = {"CHF": 1.0}
    codes = {c.upper() for c in target_codes if c} - {"CHF", "BTC"}
    for code in codes:
        # 1) Frankfurter (ECB)
        try:
            r = requests.get(
                f"https://api.frankfurter.app/latest?amount=1&from={code}&to=CHF",
                timeout=10,
            )
            r.raise_for_status()
            data = r.json().get("rates", {})
            if "CHF" in data:
                rates[code] = float(data["CHF"])
                continue
        except Exception:
            pass
        # 2) exchangerate.host fallback
        try:
            r = requests.get(
                f"https://api.exchangerate.host/convert?from={code}&to=CHF",
                timeout=10,
            )
            r.raise_for_status()
            res = r.json().get("result")
            if res is not None:
                rates[code] = float(res)
                continue
        except Exception:
            pass
    return rates



def fetch_btc_chf() -> Optional[float]:
    try:
        url = st.secrets["api"]["btc_url"]
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        j = resp.json()
        return float(j.get("bitcoin", {}).get("chf"))
    except Exception:
        return None


def compute_live_view(accts: pd.DataFrame, fx_tbl: pd.DataFrame, use_live: bool, btc_override: Optional[float]) -> pd.DataFrame:
    if accts is None or accts.empty:
        return pd.DataFrame()

    df = accts.copy()
    df["currency"] = df["currency"].astype(str).str.upper()

    # start with whatever is cached
    rate_map = {row["code"].upper(): float(row["override_rate"] or row["rate_to_chf"]) for _, row in fx_tbl.iterrows()}

    # ALWAYS refresh live FX for all codes we see (except CHF/BTC)
    codes = sorted(set(df["currency"]) - {"CHF", "BTC"})
    if use_live and codes:
        live = fetch_live_fx(codes)  # returns <code> -> CHF
        for k, v in live.items():
            rate_map[k] = v
            upsert_fx_row(k, v, source="live")

    # BTC handling (price in CHF)
    if use_live:
        live_btc = fetch_btc_chf()
        if live_btc:
            rate_map["BTC"] = live_btc
            upsert_fx_row("BTC", live_btc, source="coingecko")
    if btc_override is not None and btc_override > 0:
        rate_map["BTC"] = float(btc_override)
        upsert_fx_row("BTC", rate_map["BTC"], source="override", override=rate_map["BTC"])

    # map rates and warn if any missing
    df["rate_to_chf"] = df["currency"].map(rate_map)
    missing_codes = sorted(set(df.loc[df["rate_to_chf"].isna(), "currency"]))
    if missing_codes:
        st.warning(f"Missing FX rates for: {', '.join(missing_codes)}. Using 1.0 temporarily.")
    df["rate_to_chf"] = df["rate_to_chf"].fillna(1.0)

    # final conversion
    df["value_chf"] = (pd.to_numeric(df["value_lc"], errors="coerce").fillna(0.0) * df["rate_to_chf"]).astype(float)
    total = df["value_chf"].sum() or 1.0
    df["pct"] = df["value_chf"] / total
    return df

# -----------------------
# Pages
# -----------------------
def live_page(role: str):
    st.title("Live Portfolio")

    use_live = st.toggle("Use live FX/BTC where available", value=True, help="Overrides can still be applied below.")
    btc_override = st.number_input("BTC → CHF override (optional)", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
    btc_override = btc_override if btc_override > 0 else None

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

        col_s, col_d, col_util = st.columns([1, 1, 1])

        with col_s:
            if st.button("Save accounts"):
                clean = edited.drop(columns=[c for c in ["created_at"] if c in edited.columns])
                # Drop any all-empty editor rows (Streamlit often adds a blank trailing row)
                req = ["type", "institution", "currency", "value_lc", "class_tag"]
                clean = clean[~clean[req].isnull().all(axis=1)]
                upsert_accounts(clean)
                st.success("Saved.")
                time.sleep(0.6)
                st.experimental_rerun()

        with col_d:
            if not accts.empty:
                to_delete = st.selectbox(
                    "Delete row by id (careful)",
                    options=["-"] + accts["id"].astype(str).tolist()
                )
                if to_delete != "-" and st.button("Confirm delete"):
                    delete_account(to_delete)
                    st.success("Deleted.")
                    time.sleep(0.5)
                    st.experimental_rerun()

        with col_util:
            st.markdown("**Utilities**")
            if st.button("Refresh FX cache (force refetch)"):
                sb.table("fx_rates").delete().neq("code", None).execute()
                st.success("FX cache cleared — toggling live FX will refetch.")
                time.sleep(0.6)
                st.experimental_rerun()

        st.divider()
        st.subheader("Import accounts from Excel (Live tab)")
        up = st.file_uploader("Upload your Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False, key="live_up")
        if up:
            try:
                xl = pd.ExcelFile(up)
                if "Live" not in xl.sheet_names:
                    st.error("Couldn't find a 'Live' sheet in this workbook.")
                else:
                    df_live = xl.parse("Live", header=None)
                    rows = []
                    # Heuristic: pull rows that look like [Type, Institution, Value(LC)...]
                    for _, row in df_live.iterrows():
                        t = str(row[0]).strip() if pd.notnull(row[0]) else ""
                        inst = str(row[1]).strip() if pd.notnull(row[1]) else ""
                        val = pd.to_numeric(row[2], errors="coerce")
                        if t.lower() in ("cash","equity","bitcoin","pension") and pd.notnull(val):
                            default_class = "Cash+Bonds" if t.lower()=="cash" else ("Global Equity" if t.lower() in ("equity","bitcoin") else "Swiss Equity")
                            rows.append({
                                "type": t.title(),
                                "institution": inst or "Unknown",
                                "currency": "CHF",  # adjust after import
                                "value_lc": float(val),
                                "class_tag": default_class,
                                "is_liquid": t.lower() in ("cash","equity","bitcoin"),
                                "notes": None
                            })
                    if rows:
                        df_rows = pd.DataFrame(rows)
                        st.dataframe(df_rows, use_container_width=True)
                        if st.button("Insert imported rows"):
                            upsert_accounts(df_rows)
                            st.success("Imported.")
                            st.experimental_rerun()
                    else:
                        st.warning("Importer didn't recognize rows. You can paste manually into the editor above.")
            except Exception as e:
                st.error(f"Import failed: {e}")

    # After any edits/imports, compute view
    fx_tbl = get_fx_table()
    live_df = compute_live_view(get_accounts_df(), fx_tbl, use_live=use_live, btc_override=btc_override)

    st.subheader("Portfolio breakdown (CHF)")
    if live_df.empty:
        st.info("Add accounts to see charts.")
        return

    by_inst = live_df.groupby("institution", as_index=False)["value_chf"].sum().sort_values("value_chf", ascending=False)
    fig_bar = px.bar(by_inst, x="institution", y="value_chf", title="By Institution", color_discrete_sequence=BRIGHT)
    st.plotly_chart(fig_bar, use_container_width=True)

    by_class = live_df.groupby("class_tag", as_index=False)["value_chf"].sum()
    fig_pie = px.pie(by_class, names="class_tag", values="value_chf", hole=0.55, title="Allocation by Class", color_discrete_sequence=BRIGHT)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Totals")
    total = live_df["value_chf"].sum()
    liquid = live_df[live_df["is_liquid"]]["value_chf"].sum()
    c1, c2, _ = st.columns(3)
    c1.metric("Grand Total (CHF)", f"{total:,.2f}")
    c2.metric("Liquid Assets (CHF)", f"{liquid:,.2f}")

    st.subheader("Detail table")
    show = live_df[["type","institution","currency","value_lc","rate_to_chf","value_chf","class_tag","is_liquid"]].copy()
    show = show.sort_values("value_chf", ascending=False)
    st.dataframe(show, use_container_width=True)


def projection_page(role: str):
    st.title("Projections (hard-coded schedule)")

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
                        # Find first row that looks like it starts data (heuristic)
                        start_row = None
                        for r in range(min(10, len(raw))):
                            row = raw.iloc[r].tolist()
                            if any(isinstance(v, (dt.date, dt.datetime, pd.Timestamp)) for v in row):
                                start_row = r
                                break
                        if start_row is None:
                            start_row = 2
                        data = raw.iloc[start_row:].dropna(how="all", axis=1)
                        data.columns = range(data.shape[1])
                        cols_map = {0:"dt",1:"cash",2:"bitcoin",3:"pillar3a",4:"pillar2",5:"ibkr",6:"pillar1e"}
                        out = data[list(cols_map.keys())].rename(columns=cols_map)
                        out = out[out["dt"].notnull()]
                        out["dt"] = pd.to_datetime(out["dt"]).dt.date
                        for c in ["cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e"]:
                            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
                        out["grand_total"] = out[["cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e"]].sum(axis=1)
                        st.dataframe(out.head(), use_container_width=True)
                        if st.button("Replace projection table with imported data"):
                            sb.table("projection_rows").delete().neq("dt", None).execute()
                            upsert_projection_df(out)
                            st.success("Projection imported.")
                            st.experimental_rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    st.subheader("Schedule")
    if role == ROLE_VIEWER:
        st.dataframe(df, use_container_width=True)
    else:
        edited = st.data_editor(
            df if not df.empty else pd.DataFrame(columns=["dt","cash","bitcoin","pillar3a","pillar2","ibkr","pillar1e","grand_total"]),
            num_rows="dynamic",
            hide_index=True,
            use_container_width=True,
            column_config={
                "dt": st.column_config.DateColumn("Date", format="YYYY-MM-DD")
            },
            key="proj_editor",
        )
        if st.button("Save schedule"):
            upsert_projection_df(edited)
            st.success("Saved.")
            time.sleep(0.5)
            st.experimental_rerun()

    df = get_projection_df()
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

page = st.sidebar.radio("Navigate", ["Live", "Projections"], index=0)
if page == "Live":
    live_page(role)
else:
    projection_page(role)

"""
Two-source sediment unmixing tool (Streamlit)

Enter geochemistry (e.g. XRF) for two upstream catchments (A, B) and one
sediment sample downstream of the confluence (M). The app estimates the share
of M that comes from A and B, and runs a set of checks to help you judge
whether the result makes sense.

Run with:   streamlit run unmixing_tool.py
Requires:   streamlit numpy pandas matplotlib
"""
import io
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sediment unmixing tool", layout="wide")

COLS = ["Use", "Element", "Unit", "A", "A_sd", "B", "B_sd", "M", "M_sd"]
WEIGHTING = {
    "Inverse variance (uses your uncertainties)": "ivw",
    "Relative error (ignores uncertainties)": "rel",
}
OXIDE = re.compile(r"^[A-Z][a-z]?\d*O\d*$")      # CaO, SiO2, Al2O3, K2O, ...


# ----------------------------------------------------------------------
# Example and template data
# ----------------------------------------------------------------------
def example_df():
    el = ["CaO", "SiO2", "Al2O3", "Fe2O3", "K2O", "TiO2", "Sr", "Zr", "Rb"]
    unit = ["wt%"] * 6 + ["ppm"] * 3
    A = np.array([36.55, 18.11, 4.66, 2.35, 0.90, 0.25, 302.8, 59.1, 28.1])
    B = np.array([1.17, 82.72, 5.59, 1.57, 1.60, 0.42, 44.6, 247.2, 62.9])
    M = np.array([13.05, 60.14, 5.06, 1.80, 1.40, 0.38, 140.6, 230.1, 47.6])
    return pd.DataFrame({
        "Use": True, "Element": el, "Unit": unit,
        "A": A, "A_sd": np.round(0.08 * A, 3),
        "B": B, "B_sd": np.round(0.08 * B, 3),
        "M": M, "M_sd": np.round(0.04 * M, 3),
    })[COLS]


def blank_df(n=6):
    return pd.DataFrame({
        "Use": [True] * n, "Element": [""] * n, "Unit": [""] * n,
        **{c: [np.nan] * n for c in ["A", "A_sd", "B", "B_sd", "M", "M_sd"]},
    })[COLS]


# ----------------------------------------------------------------------
# CSV parsing
# ----------------------------------------------------------------------
def guess_unit(el):
    return "wt%" if OXIDE.match(str(el)) else "ppm"


def parse_upload(file):
    """Accept either a summary table or replicate samples (wide format).

    Summary:    columns Element, A, B, M (+ optional Unit, A_sd, B_sd, M_sd)
    Replicates: column 'group' (A / B / M) + one column per element,
                one row per sample.
    Returns (df, nA, nB, nM) where n's are None for the summary format.
    """
    raw = pd.read_csv(file, sep=None, engine="python")
    raw.columns = [str(c).strip() for c in raw.columns]
    low = {c.lower(): c for c in raw.columns}

    if "group" in low:                                       # replicate format
        g = raw[low["group"]].astype(str).str.strip().str.upper()
        g = g.replace({"MIX": "M", "MIXTURE": "M", "MIXED": "M"})
        skip = {low["group"]} | {low[k] for k in ("sample", "id", "sample_id")
                                 if k in low}
        elems = [c for c in raw.columns if c not in skip]
        num = raw[elems].apply(pd.to_numeric, errors="coerce")
        rows, ns = [], {}
        for grp in ("A", "B", "M"):
            if not (g == grp).any():
                raise ValueError(f"No rows with group '{grp}'.")
            sub = num[g == grp]
            ns[grp] = int(sub.count().max())
        for e in elems:
            row = dict(Use=True, Element=e, Unit=guess_unit(e))
            for grp in ("A", "B", "M"):
                s = num.loc[g == grp, e]
                row[grp] = s.mean()
                row[grp + "_sd"] = s.std(ddof=1) if s.count() > 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows)[COLS], ns["A"], ns["B"], ns["M"]

    need = {"element", "a", "b", "m"}                        # summary format
    if not need.issubset(low):
        raise ValueError("Need columns Element, A, B, M (summary) or a 'group' "
                         "column plus one column per element (replicates).")
    out = pd.DataFrame({"Use": True, "Element": raw[low["element"]]})
    out["Unit"] = raw[low["unit"]] if "unit" in low else out["Element"].map(guess_unit)
    for c in ["A", "A_sd", "B", "B_sd", "M", "M_sd"]:
        out[c] = pd.to_numeric(raw[low[c.lower()]], errors="coerce") \
            if c.lower() in low else np.nan
    return out[COLS], None, None, None


# ----------------------------------------------------------------------
# Core maths
# ----------------------------------------------------------------------
def prepare(df, nA, nB, nM, cv_default):
    d = df.copy()
    d["Element"] = d["Element"].astype(str).str.strip()
    d = d[(d["Element"] != "") & (d["Element"].str.lower() != "nan")]
    for c in ["A", "A_sd", "B", "B_sd", "M", "M_sd"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["A", "B", "M"]).reset_index(drop=True)
    d["Use"] = d["Use"].fillna(True).astype(bool)
    d["Unit"] = d["Unit"].fillna("").astype(str)
    assumed = False
    for v, n in (("A", nA), ("B", nB), ("M", nM)):
        sd = d[v + "_sd"].where(d[v + "_sd"] > 0)
        if sd.isna().any():
            assumed = True
        sd = sd.fillna(cv_default * d[v].abs())
        d[v + "_se"] = sd / np.sqrt(max(int(n), 1))
    return d, assumed


def analyse(d, weighting, thr, auto, n_mc, seed=1):
    A, B, M = d["A"].values, d["B"].values, d["M"].values
    A_se, B_se, M_se = d["A_se"].values, d["B_se"].values, d["M_se"].values
    names = d["Element"].tolist()
    dd, rr = A - B, M - B

    def variance(f):
        return np.maximum(f**2 * A_se**2 + (1 - f)**2 * B_se**2 + M_se**2, 1e-30)

    def fit(mask):
        if weighting == "rel":
            w = 1.0 / np.maximum(M, 1e-12) ** 2
            den = np.sum(w[mask] * dd[mask] ** 2)
            if den <= 0:
                raise ValueError("The chosen elements do not discriminate A from B.")
            raw = np.sum(w[mask] * dd[mask] * rr[mask]) / den
            return float(np.clip(raw, 0, 1)), float(raw), w
        f = 0.5
        for _ in range(30):
            w = 1.0 / variance(f)
            den = np.sum(w[mask] * dd[mask] ** 2)
            if den <= 0:
                raise ValueError("The chosen elements do not discriminate A from B.")
            raw = np.sum(w[mask] * dd[mask] * rr[mask]) / den
            f_new = float(np.clip(raw, 0, 1))
            if abs(f_new - f) < 1e-10:
                f = f_new
                break
            f = f_new
        return f, float(raw), 1.0 / variance(f)

    def zscores(f):
        return (M - (f * A + (1 - f) * B)) / np.sqrt(variance(f))

    mask = d["Use"].values.copy()
    if mask.sum() < 2:
        raise ValueError("Select at least 2 elements.")
    dropped = []
    while auto:
        f, raw, w = fit(mask)
        z = np.where(mask, np.abs(zscores(f)), 0.0)
        j = int(z.argmax())
        if z[j] > thr and mask.sum() > 3:
            mask[j] = False
            dropped.append(names[j])
        else:
            break

    f, raw, w = fit(mask)
    z = zscores(f)
    pred = f * A + (1 - f) * B
    n_used = int(mask.sum())
    chi2 = float(np.sum(z[mask] ** 2))
    red_chi2 = chi2 / max(n_used - 1, 1)

    # Monte Carlo uncertainty
    rng = np.random.default_rng(seed)
    shape = (n_mc, len(A))
    Ak = rng.normal(A, A_se, shape)[:, mask]
    Bk = rng.normal(B, B_se, shape)[:, mask]
    Mk = rng.normal(M, M_se, shape)[:, mask]
    dk, rk, wm = Ak - Bk, Mk - Bk, w[mask]
    fs = np.clip((wm * dk * rk).sum(1) / np.maximum((wm * dk * dk).sum(1), 1e-30), 0, 1)
    lo, med, hi = np.percentile(fs, [2.5, 50, 97.5])

    # Leave-one-out sensitivity
    loo = {}
    if n_used >= 3:
        for i in np.where(mask)[0]:
            m2 = mask.copy()
            m2[i] = False
            try:
                loo[names[i]] = fit(m2)[0]
            except ValueError:
                pass

    # Per-element diagnostics
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = rr / dd
        implied_se = np.sqrt(M_se**2 / dd**2 + rr**2 / dd**4 * A_se**2
                             + (M - A) ** 2 / dd**4 * B_se**2)
    contrast = np.abs(dd) / np.sqrt(A_se**2 + B_se**2)
    lo_b, hi_b = np.minimum(A, B), np.maximum(A, B)
    bracketed = (M >= lo_b - 2 * M_se) & (M <= hi_b + 2 * M_se)

    # Closure of wt% totals
    wt = d["Unit"].str.lower().str.replace(" ", "").isin(["wt%", "%", "wt.%"]).values
    closure = None
    if wt.sum() >= 3:
        sA, sB, sM = A[wt].sum(), B[wt].sum(), M[wt].sum()
        closure = dict(A=sA, B=sB, M=sM, pred=f * sA + (1 - f) * sB,
                       diff=sM - (f * sA + (1 - f) * sB))

    table = pd.DataFrame({
        "Element": names, "Unit": d["Unit"], "Used": mask,
        "A": A, "B": B, "M": M, "Predicted M": pred, "Residual": M - pred,
        "z (residual/σ)": z, "Contrast A–B (SE)": contrast,
        "M between A and B": bracketed, "Implied share of A": implied,
    })
    return dict(f=f, raw=raw, lo=lo, med=med, hi=hi, fs=fs, mask=mask,
                dropped=dropped, z=z, pred=pred, red_chi2=red_chi2, n_used=n_used,
                loo=loo, implied=implied, implied_se=implied_se, contrast=contrast,
                bracketed=bracketed, closure=closure, table=table, names=names,
                A=A, B=B, M=M, A_se=A_se, B_se=B_se, M_se=M_se)


def run_checks(o, thr, nA, nB, nM, assumed):
    """Return a list of (status, title, detail) and an overall verdict."""
    c, mask = [], o["mask"]
    names = np.array(o["names"])

    ng = int((o["contrast"][mask] > 3).sum())
    c.append(("ok" if ng >= 3 else "warn" if ng == 2 else "bad",
              "Discriminating elements",
              f"{ng} of the {o['n_used']} used elements differ between A and B by "
              f"more than 3 standard errors. Aim for at least 3."))

    bad_b = names[mask & ~o["bracketed"]].tolist()
    c.append(("ok" if not bad_b else "warn" if len(bad_b) == 1 else "bad",
              "Mixture lies between A and B",
              "All used elements are bracketed." if not bad_b else
              f"Not bracketed: {', '.join(bad_b)}. A simple two-source mixture "
              "cannot produce these values. Look for a third source, grain-size "
              "sorting or chemical change."))

    r = o["red_chi2"]
    if o["n_used"] >= 3:
        c.append(("ok" if r <= 1.5 else "warn" if r <= 3 else "bad",
                  "Consistency of the elements (reduced χ²)",
                  f"{r:.2f}. Around 1 means the elements agree within their "
                  "uncertainties. Much larger means they tell different stories "
                  "(or your uncertainties are too small)."))
    else:
        c.append(("info", "Consistency of the elements",
                  "Needs at least 3 used elements to be meaningful."))

    out_el = names[mask & (np.abs(o["z"]) > thr)].tolist()
    c.append(("ok" if not out_el else "warn", "Outlier elements",
              f"No used element deviates by more than {thr:g}σ." if not out_el
              else f"Beyond {thr:g}σ: {', '.join(out_el)}."))
    if o["dropped"]:
        c.append(("info", "Auto-excluded elements",
                  f"Removed automatically: {', '.join(o['dropped'])}."))

    raw = o["raw"]
    c.append(("ok" if -0.02 <= raw <= 1.02 else "warn" if -0.10 <= raw <= 1.10 else "bad",
              "Unconstrained estimate inside 0-1",
              f"Before clipping, the fit gives {raw:.3f}. Values well outside 0-1 "
              "mean the mixture is not explained by A and B alone."))

    w = o["hi"] - o["lo"]
    c.append(("ok" if w < 0.15 else "warn" if w < 0.35 else "bad",
              "Width of the 95 % interval",
              f"{o['lo']:.1%} to {o['hi']:.1%} (width {w:.1%})."))

    if o["loo"]:
        dmax = max(abs(v - o["f"]) for v in o["loo"].values())
        who = max(o["loo"], key=lambda k: abs(o["loo"][k] - o["f"]))
        c.append(("ok" if dmax < 0.05 else "warn" if dmax < 0.10 else "bad",
                  "Stability when dropping one element",
                  f"The largest change is {dmax:.3f}, when dropping {who}."))

    cl = o["closure"]
    if cl:
        dd = abs(cl["diff"])
        c.append(("ok" if dd < 3 else "warn" if dd < 6 else "bad",
                  "Closure of wt % totals",
                  f"Sums: A {cl['A']:.1f}, B {cl['B']:.1f}, M {cl['M']:.1f}; mixing "
                  f"predicts {cl['pred']:.1f} (difference {cl['diff']:+.1f}). "
                  "Only meaningful if your wt % list is close to complete."))

    if min(nA, nB, nM) == 1:
        c.append(("info", "Single measurements",
                  "With n = 1 in a group, natural variability is not captured. "
                  "The interval reflects only the uncertainties you entered."))
    if assumed:
        c.append(("info", "Assumed uncertainties",
                  "Some SDs were missing and replaced by the default relative "
                  "error from the sidebar."))

    statuses = [s for s, _, _ in c]
    if "bad" in statuses:
        verdict = ("error", "Treat this result with caution: at least one check failed.")
    elif "warn" in statuses:
        verdict = ("warning", "Plausible, but some checks raise warnings.")
    else:
        verdict = ("success", "The data are consistent with a two-source mixture.")
    return c, verdict


# ===== UI ==============================================================
ICON = {"ok": "✅", "warn": "⚠️", "bad": "❌", "info": "ℹ️"}


def _init():
    ss = st.session_state
    ss.setdefault("tbl", example_df())
    ss.setdefault("ver", 0)
    ss.setdefault("nA", 4)
    ss.setdefault("nB", 4)
    ss.setdefault("nM", 3)
    ss.setdefault("msg", None)


def load_example():
    st.session_state.update(tbl=example_df(), ver=st.session_state.ver + 1,
                            nA=4, nB=4, nM=3, msg=None)


def clear_table():
    st.session_state.update(tbl=blank_df(), ver=st.session_state.ver + 1,
                            nA=1, nB=1, nM=1, msg=None)


def on_upload():
    f = st.session_state.get("upload")
    if f is None:
        return
    try:
        df, nA, nB, nM = parse_upload(f)
        st.session_state.tbl = df
        st.session_state.ver += 1
        if nA is not None:
            st.session_state.nA, st.session_state.nB, st.session_state.nM = nA, nB, nM
        st.session_state.msg = ("ok", f"Loaded {len(df)} elements from {f.name}.")
    except Exception as e:  # noqa: BLE001
        st.session_state.msg = ("err", f"Could not read the file: {e}")


_init()

with st.sidebar:
    st.header("Data")
    st.file_uploader("Upload CSV", type=["csv", "txt"], key="upload",
                     on_change=on_upload,
                     help="Summary table (Element, A, B, M, optional Unit and "
                          "A_sd, B_sd, M_sd) or replicate samples (column "
                          "'group' = A/B/M plus one column per element).")
    st.download_button("Download CSV template",
                       example_df().drop(columns="Use").to_csv(index=False),
                       "unmixing_template.csv", "text/csv")
    c1, c2 = st.columns(2)
    c1.button("Load example", on_click=load_example)
    c2.button("Clear table", on_click=clear_table)

    st.header("Samples")
    st.text_input("Name of catchment A", "A", key="nameA")
    st.text_input("Name of catchment B", "B", key="nameB")
    st.number_input("Replicates behind A values", 1, 500, key="nA",
                    help="Standard error = SD / √n. Use 1 if the SD is already "
                         "the uncertainty of the value.")
    st.number_input("Replicates behind B values", 1, 500, key="nB")
    st.number_input("Replicates behind M values", 1, 500, key="nM")

    st.header("Method")
    wname = st.selectbox("Weighting", list(WEIGHTING))
    cv = st.slider("Default relative error where SD is missing (%)", 1, 30, 5) / 100
    auto = st.checkbox("Auto-exclude outlier elements", value=False)
    thr = st.slider("Outlier threshold (σ)", 2.0, 5.0, 3.0, 0.5)
    n_mc = st.select_slider("Monte Carlo iterations", [1000, 3000, 5000, 10000], 3000)

nmA = st.session_state.nameA.strip() or "A"
nmB = st.session_state.nameB.strip() or "B"

st.title("Sediment unmixing tool")
st.write(
    f"Estimate how much of a mixed sediment comes from **{nmA}** and **{nmB}** "
    "from their geochemistry, and check whether the answer makes sense."
)
with st.expander("How it works and how to use it"):
    st.markdown(
        "**Model.** For each element, M = f·A + (1−f)·B, where f is the share of "
        "the mixed sediment that comes from A. All elements are combined in a "
        "weighted least-squares fit, with f limited to 0-1. Uncertainty comes "
        "from a Monte Carlo simulation using the standard errors.\n\n"
        "**Input.** Enter values in the table, or upload a CSV. Use the same "
        "units, grain-size fraction and analytical method for A, B and M. "
        "Leave an SD blank to use the default relative error. Untick **Use** "
        "to exclude an element.\n\n"
        "**Important.** The result is the share of *sediment mass as seen "
        "through the chosen tracers*. It does not measure erosion rates or water "
        "discharge, and it needs A and B to be the only sources."
    )

if st.session_state.msg:
    kind, text = st.session_state.msg
    (st.success if kind == "ok" else st.error)(text)

st.subheader("Geochemistry")
st.caption("Values are means; SD is the standard deviation of replicates "
           "(or leave blank). Add rows with the + at the bottom of the table.")
edited = st.data_editor(
    st.session_state.tbl, key=f"editor_{st.session_state.ver}",
    num_rows="dynamic", hide_index=True,
    column_config={
        "Use": st.column_config.CheckboxColumn("Use", default=True),
        "Element": st.column_config.TextColumn("Element"),
        "Unit": st.column_config.TextColumn("Unit", help="Use 'wt%' for oxides "
                                            "to enable the closure check."),
        "A": st.column_config.NumberColumn(nmA, format="%.4g"),
        "A_sd": st.column_config.NumberColumn(f"{nmA} SD", format="%.4g"),
        "B": st.column_config.NumberColumn(nmB, format="%.4g"),
        "B_sd": st.column_config.NumberColumn(f"{nmB} SD", format="%.4g"),
        "M": st.column_config.NumberColumn("Mixture", format="%.4g"),
        "M_sd": st.column_config.NumberColumn("Mixture SD", format="%.4g"),
    },
)

d, assumed = prepare(edited, st.session_state.nA, st.session_state.nB,
                     st.session_state.nM, cv)
if (d[["A", "B", "M"]] < 0).any().any():
    st.error("Concentrations must not be negative.")
    st.stop()
if len(d) < 2:
    st.info("Enter at least two elements with values for A, B and the mixture.")
    st.stop()

try:
    o = analyse(d, WEIGHTING[wname], thr, auto, n_mc)
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------- Results ----------------
st.divider()
st.header("Result")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Share from {nmA}", f"{o['f']:.1%}")
m2.metric(f"Share from {nmB}", f"{1 - o['f']:.1%}")
m3.metric("95 % interval (A)", f"{o['lo']:.1%} – {o['hi']:.1%}")
m4.metric("Elements used", f"{o['n_used']} of {len(d)}")

checks, (vkind, vtext) = run_checks(o, thr, st.session_state.nA,
                                    st.session_state.nB, st.session_state.nM,
                                    assumed)
st.subheader("Do the results make sense?")
getattr(st, vkind)(vtext)
for status, title, detail in checks:
    st.markdown(f"{ICON[status]} **{title}.** {detail}")

# ---------------- Plots ----------------
st.subheader("Diagnostics")
t1, t2, t3, t4, t5 = st.tabs(["Fit", "Per-element share", "Sensitivity",
                              "Uncertainty", "Table"])
mask = o["mask"]
names = o["names"]
col = np.where(mask, "tab:green", "tab:red")

with t1:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    pos = (o["pred"] > 0) & (o["M"] > 0)
    ax[0].errorbar(o["pred"][pos], o["M"][pos], yerr=o["M_se"][pos], fmt="none",
                   ecolor="gray", lw=1)
    for x, y, n, u in zip(o["pred"][pos], o["M"][pos], np.array(names)[pos], mask[pos]):
        ax[0].plot(x, y, "o", color="tab:green" if u else "tab:red")
        ax[0].annotate(n, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    lim = [min(o["pred"][pos].min(), o["M"][pos].min()) * 0.7,
           max(o["pred"][pos].max(), o["M"][pos].max()) * 1.4]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Predicted mixture"); ax[0].set_ylabel("Measured mixture")
    ax[0].set_title("Measured vs predicted (green = used)")
    ax[1].bar(names, o["z"], color=col)
    for s in (-thr, thr):
        ax[1].axhline(s, color="gray", ls="--", lw=1)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("Standardised residual (σ)")
    ax[1].set_title("Residuals")
    ax[1].tick_params(axis="x", rotation=45)
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

with t2:
    fig, ax = plt.subplots(figsize=(8, 4))
    ok = np.isfinite(o["implied"]) & np.isfinite(o["implied_se"])
    ax.bar(np.array(names)[ok], np.clip(o["implied"][ok], -0.5, 1.5),
           yerr=np.clip(o["implied_se"][ok], 0, 1), color=col[ok], capsize=3)
    ax.axhline(o["f"], color="k", ls="--", label="estimate")
    ax.axhspan(o["lo"], o["hi"], color="k", alpha=0.08, label="95 % interval")
    ax.set_ylim(-0.5, 1.5)
    ax.set_ylabel(f"Implied share from {nmA}")
    ax.set_title("What each element says on its own (red = excluded)")
    ax.tick_params(axis="x", rotation=45); ax.legend()
    fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("If the mixture is a clean blend, all bars agree. Elements with "
               "little contrast between A and B have large error bars and add "
               "little information. Values are clipped to -0.5..1.5 for display.")

with t3:
    if o["loo"]:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.bar(list(o["loo"]), list(o["loo"].values()), color="tab:blue")
        ax.axhline(o["f"], color="k", ls="--", label="all used elements")
        ax.set_ylabel(f"Share from {nmA}")
        ax.set_title("Estimate when each element is dropped")
        ax.tick_params(axis="x", rotation=45); ax.legend()
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    else:
        st.info("Use at least 3 elements to see leave-one-out results.")

with t4:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(o["fs"], bins=40, color="tab:orange", alpha=0.85)
    ax.axvline(o["med"], color="k", ls="--", label="median")
    ax.axvspan(o["lo"], o["hi"], color="k", alpha=0.1, label="95 %")
    ax.set_xlabel(f"Share from {nmA}")
    ax.set_title("Monte Carlo distribution")
    ax.legend(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption("This captures the uncertainties you entered. It does not "
               "capture missing sources, grain-size effects or non-conservative "
               "behaviour.")

with t5:
    show = o["table"].copy()
    st.dataframe(show.round(4), hide_index=True)
    st.download_button("Download per-element table (CSV)",
                       show.to_csv(index=False), "unmixing_elements.csv",
                       "text/csv")
    summary = pd.DataFrame({
        "quantity": ["share_A", "share_B", "ci95_low_A", "ci95_high_A",
                     "reduced_chi2", "elements_used"],
        "value": [o["f"], 1 - o["f"], o["lo"], o["hi"], o["red_chi2"], o["n_used"]]})
    st.download_button("Download summary (CSV)", summary.to_csv(index=False),
                       "unmixing_summary.csv", "text/csv")

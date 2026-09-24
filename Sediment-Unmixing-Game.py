"""
Sediment Unmixing Game  (Streamlit)

Two upstream catchments (limestone vs sandstone) merge; you get XRF data for
both and for the mixed sediment downstream. Guess the limestone share, pick
the elements you trust, run the unmixing and see how close you get.

All data are SYNTHETIC, based on typical fine-fraction stream-sediment
compositions (major oxides in wt %, trace elements in ppm).

Run with:   streamlit run unmixing_game.py
Requires:   streamlit numpy pandas matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ----------------------------------------------------------------------
# Data and settings
# ----------------------------------------------------------------------
ELEMENTS = ["CaO", "SiO2", "Al2O3", "Fe2O3", "K2O", "TiO2", "Sr", "Zr", "Rb"]
UNITS = ["wt%"] * 6 + ["ppm"] * 3
A_TRUE = np.array([36.0, 18.0, 4.5, 2.2, 0.9, 0.25, 320.0, 60.0, 28.0])   # limestone
B_TRUE = np.array([1.2, 84.0, 5.5, 1.6, 1.6, 0.45, 45.0, 260.0, 62.0])    # sandstone

LEVELS = {
    "Easy":   dict(nat=0.05, ana=0.02, n_rep=5, n_mix=3),
    "Medium": dict(nat=0.08, ana=0.03, n_rep=4, n_mix=3),
    "Hard":   dict(nat=0.20, ana=0.05, n_rep=3, n_mix=3),
}
TRAPS = {
    "Zr": "heavy-mineral (zircon) sorting in the mixed sediment",
    "Sr": "Sr gained or lost through carbonate dissolution and reprecipitation",
    "CaO": "carbonate dissolution between the confluence and the sample site",
    "K2O": "mica winnowing during transport",
}
MIN_ELEMENTS = 3


# ----------------------------------------------------------------------
# Simulation and unmixing
# ----------------------------------------------------------------------
def simulate(seed, level, trap_on):
    p = LEVELS[level]
    rng = np.random.default_rng(seed)
    f_true = float(rng.uniform(0.10, 0.90))

    def reps(true, n):
        nat = rng.normal(1.0, p["nat"], (n, true.size))
        ana = rng.normal(1.0, p["ana"], (n, true.size))
        return true * nat * ana

    A_reps = reps(A_TRUE, p["n_rep"])
    B_reps = reps(B_TRUE, p["n_rep"])

    mix_true = f_true * A_TRUE + (1 - f_true) * B_TRUE
    trap = None
    if trap_on:
        el = str(rng.choice(list(TRAPS)))
        factor = float(rng.choice([rng.uniform(1.25, 1.45), rng.uniform(0.60, 0.78)]))
        mix_true[ELEMENTS.index(el)] *= factor
        trap = dict(element=el, factor=factor, reason=TRAPS[el])
    M_reps = reps(mix_true, p["n_mix"])

    def stats(x):
        return x.mean(0), x.std(0, ddof=1), x.std(0, ddof=1) / np.sqrt(x.shape[0])

    A, A_sd, A_se = stats(A_reps)
    B, B_sd, B_se = stats(B_reps)
    M, M_sd, M_se = stats(M_reps)
    return dict(seed=seed, level=level, f_true=f_true, trap=trap, done=False,
                A=A, A_sd=A_sd, A_se=A_se, B=B, B_sd=B_sd, B_se=B_se,
                M=M, M_sd=M_sd, M_se=M_se)


def weights(R):
    return 1.0 / (R["A_se"] ** 2 + R["B_se"] ** 2 + R["M_se"] ** 2)


def solve_f(A, B, M, w):
    d, r = A - B, M - B
    return float(np.clip(np.sum(w * d * r) / np.sum(w * d * d), 0.0, 1.0))


def diagnostics(R):
    disc = np.abs(R["A"] - R["B"]) / np.sqrt(R["A_se"] ** 2 + R["B_se"] ** 2)
    lo, hi = np.minimum(R["A"], R["B"]), np.maximum(R["A"], R["B"])
    bracketed = (R["M"] >= lo - 2 * R["M_se"]) & (R["M"] <= hi + 2 * R["M_se"])
    implied_f = (R["M"] - R["B"]) / (R["A"] - R["B"])
    return disc, bracketed, implied_f


def run_unmix(R, mask, n_mc=3000):
    w = weights(R)
    f = solve_f(R["A"][mask], R["B"][mask], R["M"][mask], w[mask])
    pred = f * R["A"] + (1 - f) * R["B"]
    sigma = np.sqrt(1.0 / w)
    z = (R["M"] - pred) / sigma

    rng = np.random.default_rng(R["seed"] + 1)
    shape = (n_mc, len(ELEMENTS))
    Ak = rng.normal(R["A"], R["A_se"], shape)[:, mask]
    Bk = rng.normal(R["B"], R["B_se"], shape)[:, mask]
    Mk = rng.normal(R["M"], R["M_se"], shape)[:, mask]
    d, r, wm = Ak - Bk, Mk - Bk, w[mask]
    fs = np.clip((wm * d * r).sum(1) / (wm * d * d).sum(1), 0, 1)
    lo, med, hi = np.percentile(fs, [2.5, 50, 97.5])
    return dict(f=f, z=z, fs=fs, lo=lo, med=med, hi=hi)


def score_round(R, guess, est, mask, hint_used):
    """Points: intuition (30) + unmixing accuracy (70) + choices (+15/-3)."""
    intuition = max(0.0, 30 - 100 * abs(guess - R["f_true"]))
    accuracy = max(0.0, 70 - 700 * abs(est - R["f_true"]))
    bonus = 0.0
    if R["trap"]:
        trap_idx = ELEMENTS.index(R["trap"]["element"])
        if not mask[trap_idx]:
            bonus += 15
        bonus -= 3 * int((~mask).sum() - (0 if mask[trap_idx] else 1))
    else:
        bonus -= 3 * int((~mask).sum())
    hint = -10 if hint_used else 0
    total = max(0.0, intuition + accuracy + bonus + hint)
    return dict(intuition=intuition, accuracy=accuracy, bonus=bonus,
                hint=hint, total=total)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
def new_round():
    seed = int(np.random.default_rng().integers(0, 10**9))
    st.session_state.R = simulate(seed, st.session_state.level,
                                  st.session_state.trap_on)
    st.session_state.guess = 50
    st.session_state.elems = list(ELEMENTS)
    st.session_state.hint = False


if "history" not in st.session_state:
    st.session_state.history = []
if "level" not in st.session_state:
    st.session_state.level = "Medium"
if "trap_on" not in st.session_state:
    st.session_state.trap_on = True
if "R" not in st.session_state:
    new_round()

R = st.session_state.R

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.selectbox("Difficulty", list(LEVELS), key="level", on_change=new_round)
    st.checkbox("Hidden non-conservative element", key="trap_on",
                on_change=new_round,
                help="One element in the mixed sample is altered by a process "
                     "other than mixing. Spot it and exclude it.")
    st.button("New round", on_click=new_round)
    st.divider()
    st.header("Scoreboard")
    total = sum(h["total"] for h in st.session_state.history)
    st.metric("Total points", f"{total:.0f}")
    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history)[
                ["level", "true_f", "est_f", "total"]
            ].rename(columns={"true_f": "true", "est_f": "estimate"}),
            hide_index=True)

# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
st.title("Sediment unmixing game")
st.write(
    "A **limestone** catchment and a **sandstone** catchment join. You have XRF "
    "data for both upstream sediments and for the mixed sediment downstream. "
    "What share of the mixture comes from the limestone catchment?"
)
with st.expander("How to play"):
    st.markdown(
        "1. Look at the data and **guess** the limestone share.\n"
        "2. **Choose the elements** you trust. Some may not behave like a "
        "simple mixture (sorting, dissolution).\n"
        "3. Press **Run unmixing**. Scoring: intuition (max 30), unmixing "
        "accuracy (max 70), +15 for excluding the hidden non-conservative "
        "element, -3 for each innocent element you drop, -10 for using the hint.\n\n"
        "The maths is a weighted least-squares fit of M = f·A + (1-f)·B, with "
        "Monte Carlo uncertainty from the replicate scatter."
    )

table = pd.DataFrame({
    "Element": [f"{e} ({u})" for e, u in zip(ELEMENTS, UNITS)],
    "Limestone A": [f"{m:.2f} ± {s:.2f}" for m, s in zip(R["A"], R["A_sd"])],
    "Sandstone B": [f"{m:.2f} ± {s:.2f}" for m, s in zip(R["B"], R["B_sd"])],
    "Mixture": [f"{m:.2f} ± {s:.2f}" for m, s in zip(R["M"], R["M_sd"])],
})
p = LEVELS[R["level"]]
st.subheader("Data (mean ± SD of replicates)")
st.caption(f"{p['n_rep']} replicates per upstream catchment, "
           f"{p['n_mix']} of the mixed sediment.")
st.dataframe(table, hide_index=True)

st.subheader("1. Your guess")
guess_pct = st.slider("Limestone share of the mixed sediment (%)", 0, 100,
                      key="guess", disabled=R["done"])

st.subheader("2. Choose your elements")
chosen = st.multiselect("Elements used in the unmixing", ELEMENTS, key="elems",
                        disabled=R["done"])

hint = st.checkbox("Show hint (-10 points)", key="hint", disabled=R["done"])
if hint:
    disc, bracketed, implied_f = diagnostics(R)
    st.dataframe(pd.DataFrame({
        "Element": ELEMENTS,
        "A vs B contrast (in SE)": np.round(disc, 1),
        "Mixture between A and B": np.where(bracketed, "yes", "NO"),
        "Implied limestone share": np.round(implied_f, 2),
    }), hide_index=True)
    st.caption("If every element were a pure mixture they would all imply the "
               "same share. Elements that disagree with the rest deserve suspicion.")

st.subheader("3. Run")
too_few = len(chosen) < MIN_ELEMENTS
if too_few:
    st.warning(f"Select at least {MIN_ELEMENTS} elements.")

if st.button("Run unmixing", type="primary", disabled=R["done"] or too_few):
    mask = np.array([e in chosen for e in ELEMENTS])
    res = run_unmix(R, mask)
    sc = score_round(R, guess_pct / 100, res["f"], mask, hint)
    R.update(done=True, mask=mask, res=res, score=sc, guess_used=guess_pct / 100)
    st.session_state.history.append(dict(
        level=R["level"], true_f=round(R["f_true"], 2),
        est_f=round(res["f"], 2), total=sc["total"]))
    st.rerun()

# ----------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------
if R["done"]:
    res, sc, mask = R["res"], R["score"], R["mask"]
    st.divider()
    st.header("Result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True limestone share", f"{R['f_true']:.1%}")
    c2.metric("Your guess", f"{R['guess_used']:.1%}")
    c3.metric("Unmixing estimate", f"{res['f']:.1%}")
    c4.metric("Round score", f"{sc['total']:.0f}")
    covered = res["lo"] <= R["f_true"] <= res["hi"]
    st.write(f"Monte Carlo 95 % interval: **{res['lo']:.1%} to {res['hi']:.1%}** "
             f"({'contains' if covered else 'misses'} the true value).")
    st.caption(f"Points: intuition {sc['intuition']:.0f} + accuracy "
               f"{sc['accuracy']:.0f} + element choices {sc['bonus']:+.0f} "
               f"+ hint {sc['hint']:+.0f}")

    if R["trap"]:
        t = R["trap"]
        direction = "enriched" if t["factor"] > 1 else "depleted"
        was_excl = not mask[ELEMENTS.index(t["element"])]
        st.info(f"Hidden trap: **{t['element']}** was {direction} in the mixture "
                f"(×{t['factor']:.2f}) by {t['reason']}. "
                f"You {'excluded' if was_excl else 'kept'} it.")
    else:
        st.info("This round had no hidden non-conservative element.")

    _, _, implied_f = diagnostics(R)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar(ELEMENTS, implied_f,
              color=["tab:green" if m else "tab:red" for m in mask])
    ax[0].axhline(R["f_true"], color="gray", ls=":", label="true")
    ax[0].axhline(res["f"], color="k", ls="--", label="estimate")
    ax[0].set_ylabel("Implied limestone share")
    ax[0].set_title("Per-element implied share (red = excluded)")
    ax[0].tick_params(axis="x", rotation=45)
    ax[0].legend()
    ax[1].hist(res["fs"], bins=40, color="tab:orange", alpha=0.85)
    ax[1].axvline(R["f_true"], color="gray", ls=":", label="true")
    ax[1].axvline(res["med"], color="k", ls="--", label="median")
    ax[1].axvspan(res["lo"], res["hi"], color="k", alpha=0.1, label="95 %")
    ax[1].set_xlabel("Limestone share")
    ax[1].set_title("Monte Carlo uncertainty")
    ax[1].legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.button("Next round", on_click=new_round, type="primary")

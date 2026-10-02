"""
Generate synthetic data for a telehealth company with an AI care chat.

Outputs (in data/raw/):
    patients.csv       one row per patient
    conversations.csv  one row per AI chat conversation

Run:
    python src/generate_data.py

The big idea
------------
The data is built so that a naive comparison is WRONG. Sicker (higher
baseline_risk) patients use the chat more, and they also have worse outcomes
(more escalations, lower resolution, lower CSAT). So "chat users look worse"
is partly who chooses to chat (self-selection), not what the chat does.
That is the confound the analysis later has to deal with.

Every number below is an assumption. They're grouped at the top as named
constants so you can point to any one of them and say what it does and why.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# WHY a fixed seed: anyone who runs the script gets the exact same CSVs, so
# results can be reproduced and checked.
SEED = 42
N_PATIENTS = 20_000

# Signups run for 18 months. Observation ends 6 months after the last signup,
# so every patient has at least 6 months in which they could have chatted.
SIGNUP_START = pd.Timestamp("2024-01-01")
SIGNUP_END = pd.Timestamp("2025-06-30")
OBSERVATION_END = pd.Timestamp("2025-12-31")

# Program mix. Weight management is the biggest line for most DTC telehealth
# companies; hair is mid-size; BP management is the smallest and most clinical.
PROGRAM_SHARES = {"weight": 0.55, "hair": 0.30, "bp": 0.15}

# Plan mix. Most people start month-to-month.
# WHY plan is independent of risk: plan is just a descriptive covariate here.
# Making it independent means the only built-in confound is risk -> chat use,
# so there's exactly one thing to explain.
PLAN_SHARES = {"monthly": 0.60, "quarterly": 0.25, "annual": 0.15}

# baseline_risk is a 0-1 score of how clinically complex the patient is at
# signup. WHY Beta distributions: they live on 0-1, are right-skewed (most
# patients are low risk, with a long tail), and the shape can differ by program.
#   weight: Beta(2, 5)   mean ~0.29
#   hair:   Beta(1.5, 8) mean ~0.16  (hair loss is low clinical risk)
#   bp:     Beta(3, 4)   mean ~0.43  (hypertension patients are sicker)
RISK_BETA = {"weight": (2.0, 5.0), "hair": (1.5, 8.0), "bp": (3.0, 4.0)}

# All risk effects are written as (risk - RISK_CENTER), so the intercepts
# below describe a "typical" patient at risk 0.3 rather than a risk-0 patient
# who barely exists.
RISK_CENTER = 0.3

# --- Chat usage (the self-selection) ---------------------------------------
# WHY two parts: many patients never open the chat, and among those who do,
# usage varies a lot. A single Poisson count can't produce both "lots of
# zeros" and "a few very heavy users", so we model:
#   1) does the patient ever use chat? (logistic in risk)
#   2) if so, how many conversations? (negative binomial, rate rising with risk)
# Risk raises BOTH. That's the self-selection.
ADOPT_INTERCEPT = -0.3   # P(adopt) ~ 43% at risk 0.3
ADOPT_RISK_COEF = 3.0    # ~29% at risk 0.1, ~70% at risk 0.7

CHATS_PER_MONTH_BASE = 0.6   # for an adopter at risk 0.3
CHATS_RISK_COEF = 1.5        # log-rate per unit of risk: risk 0.7 -> ~1.1/month
# WHY a gamma "frailty": some people are just chatty, whatever their risk.
# A gamma multiplier with mean 1 turns Poisson into a negative binomial
# (overdispersed). Shape 2 = moderately heavy tail.
CHATTINESS_GAMMA_SHAPE = 2.0

# WHY front-load conversation timing: patients ask most questions during
# onboarding and first refills. Raising a uniform draw to a power > 1 pushes
# times toward the start of each patient's window.
TIMING_SKEW = 1.5

# --- Conversation content ---------------------------------------------------
INTENTS = ["refill", "shipping", "billing", "dosing_question", "side_effects", "symptom_concern"]
CLINICAL_INTENTS = {"dosing_question", "side_effects", "symptom_concern"}

# Base intent mix for a typical (risk 0.3) patient in each program.
INTENT_BASE_PROBS = {
    #          refill ship  bill  dose  side  symptom
    "weight": [0.22, 0.18, 0.15, 0.17, 0.18, 0.10],
    "hair":   [0.25, 0.25, 0.20, 0.10, 0.15, 0.05],
    "bp":     [0.25, 0.12, 0.12, 0.18, 0.13, 0.20],
}
# How much higher risk shifts the mix toward clinical topics (multinomial logit).
INTENT_RISK_COEF = {
    "refill": 0.0, "shipping": 0.0, "billing": 0.0,
    "dosing_question": 1.0, "side_effects": 1.5, "symptom_concern": 2.5,
}

# confidence = the AI's confidence in its answer. Easy logistics questions get
# high confidence; open-ended clinical questions get low confidence.
# WHY Beta again: bounded 0-1. Concentration 12 gives realistic spread.
CONFIDENCE_MEAN = {
    "refill": 0.88, "shipping": 0.90, "billing": 0.85,
    "dosing_question": 0.75, "side_effects": 0.70, "symptom_concern": 0.60,
}
CONFIDENCE_CONCENTRATION = 12.0

# --- Escalation -------------------------------------------------------------
# WHY a hard threshold: real AI care products usually have a policy rule like
# "below X confidence, always hand off to a human". Above the threshold,
# escalation is a judgment call (the patient asks for a human, the AI flags
# something). The sharp cutoff also sets up a regression discontinuity design
# later: patients just above vs just below 0.5 are nearly identical except
# for whether they were forced to a human.
ESCALATION_CONFIDENCE_THRESHOLD = 0.50
ESC_INTERCEPT = -1.8
ESC_CONFIDENCE_COEF = 6.0     # lower confidence -> more escalation
ESC_RISK_COEF = 1.5           # sicker patients get escalated more
ESC_INTENT_BUMP = {
    "refill": 0.0, "shipping": 0.0, "billing": 0.0,
    "dosing_question": 0.5, "side_effects": 0.8, "symptom_concern": 1.5,
}

# --- Resolution -------------------------------------------------------------
# resolved = the patient's issue was resolved in this conversation (by the AI,
# or by the human it was handed to).
RES_AI_INTERCEPT = -0.5
RES_AI_CONFIDENCE_COEF = 6.0  # a confident AI answer usually resolves it
RES_HUMAN_INTERCEPT = 1.2     # humans resolve ~75% of handoffs
RES_RISK_COEF = -0.8          # sicker patients' problems are harder to close out

# --- CSAT -------------------------------------------------------------------
# CSAT is 1-5 and only some people answer the survey.
# WHY model nonresponse: people with very good or very bad experiences answer
# more often. Missing CSAT is therefore NOT random, which is a realistic trap
# for anyone who just averages the csat column.
CSAT_BASE = 3.2
CSAT_RESOLVED_EFFECT = 1.2
CSAT_ESCALATED_EFFECT = -0.4  # handoffs mean waiting
CSAT_CONFIDENCE_EFFECT = 0.5
CSAT_RISK_EFFECT = -0.5
CSAT_NOISE_SD = 0.9
CSAT_RESPONSE_INTERCEPT = -0.8   # ~31% response for a middling experience
CSAT_RESPONSE_EXTREMITY = 0.6    # rises as experience moves away from neutral


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_patients(rng):
    n = N_PATIENTS

    # Signup dates: WHY a rising ramp instead of uniform: the business is
    # growing, so later months get more signups. The weight on each day goes
    # from 1 on the first day to 2 on the last.
    signup_days = pd.date_range(SIGNUP_START, SIGNUP_END, freq="D")
    day_weights = np.linspace(1.0, 2.0, len(signup_days))
    signup_date = rng.choice(signup_days, size=n, p=day_weights / day_weights.sum())

    program = rng.choice(list(PROGRAM_SHARES), size=n, p=list(PROGRAM_SHARES.values()))
    plan = rng.choice(list(PLAN_SHARES), size=n, p=list(PLAN_SHARES.values()))

    baseline_risk = np.empty(n)
    for prog, (a, b) in RISK_BETA.items():
        mask = program == prog
        baseline_risk[mask] = rng.beta(a, b, size=mask.sum())

    return pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "signup_date": pd.to_datetime(signup_date).date,
        "program": program,
        "plan": plan,
        "baseline_risk": baseline_risk.round(3),
    })


def generate_conversations(patients, rng):
    risk = patients["baseline_risk"].to_numpy()
    centered_risk = risk - RISK_CENTER
    signup = pd.to_datetime(patients["signup_date"])

    # Exposure: how long each patient could have chatted. A patient who signed
    # up early has had more time, so we scale their expected count by it.
    # Without this, early signups would look like "heavier users" for no reason.
    exposure_days = (OBSERVATION_END - signup).dt.days.to_numpy()
    exposure_months = exposure_days / 30.44

    # Part 1: does the patient ever use chat?
    p_adopt = logistic(ADOPT_INTERCEPT + ADOPT_RISK_COEF * centered_risk)
    adopted = rng.random(len(patients)) < p_adopt

    # Part 2: how many conversations, for adopters?
    rate = CHATS_PER_MONTH_BASE * np.exp(CHATS_RISK_COEF * centered_risk)
    chattiness = rng.gamma(CHATTINESS_GAMMA_SHAPE, 1.0 / CHATTINESS_GAMMA_SHAPE, size=len(patients))
    expected = rate * exposure_months * chattiness
    # An adopter has at least one conversation by definition, so add 1.
    n_convos = np.where(adopted, 1 + rng.poisson(expected), 0)

    # One row per conversation: repeat each patient's info n_convos times.
    idx = np.repeat(np.arange(len(patients)), n_convos)
    c = pd.DataFrame({
        "patient_id": patients["patient_id"].to_numpy()[idx],
        "program": patients["program"].to_numpy()[idx],
        "risk": risk[idx],
    })
    n = len(c)
    c_risk = c["risk"].to_numpy() - RISK_CENTER

    # Timestamp: front-loaded within each patient's window, plus a random time of day.
    offset_days = exposure_days[idx] * rng.random(n) ** TIMING_SKEW
    c["started_at"] = (
        signup.to_numpy()[idx]
        + pd.to_timedelta(offset_days, unit="D")
    ).floor("min")

    # Intent: base mix per program, shifted toward clinical topics as risk rises.
    # Multinomial logit: log(base prob) + risk coef * risk, then softmax.
    risk_coefs = np.array([INTENT_RISK_COEF[i] for i in INTENTS])
    intent = np.empty(n, dtype=object)
    for prog, probs in INTENT_BASE_PROBS.items():
        mask = c["program"].to_numpy() == prog
        logits = np.log(probs) + np.outer(c_risk[mask], risk_coefs)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        # Draw one intent per row: compare a uniform draw to cumulative probs.
        u = rng.random(mask.sum())[:, None]
        intent[mask] = np.array(INTENTS)[(u > p.cumsum(axis=1)).sum(axis=1)]
    c["intent"] = intent

    # Confidence: Beta with a mean that depends on intent.
    # Beta(mean*k, (1-mean)*k) has mean `mean`; k controls how spread out it is.
    mean_conf = c["intent"].map(CONFIDENCE_MEAN).to_numpy()
    k = CONFIDENCE_CONCENTRATION
    confidence = rng.beta(mean_conf * k, (1 - mean_conf) * k)
    c["confidence"] = confidence.round(3)
    confidence = c["confidence"].to_numpy()  # use the rounded value so the threshold rule matches the CSV

    # Escalation: forced below the threshold, probabilistic above it.
    is_clinical = c["intent"].isin(CLINICAL_INTENTS).to_numpy()
    esc_logit = (
        ESC_INTERCEPT
        + ESC_CONFIDENCE_COEF * (0.75 - confidence)
        + ESC_RISK_COEF * c_risk
        + c["intent"].map(ESC_INTENT_BUMP).to_numpy()
    )
    escalated = (confidence < ESCALATION_CONFIDENCE_THRESHOLD) | (rng.random(n) < logistic(esc_logit))
    c["escalated"] = escalated

    # Escalation type: who the conversation was handed to. Empty if not escalated.
    esc_type = np.full(n, "", dtype=object)
    it = c["intent"].to_numpy()
    u = rng.random(n)
    esc_type[escalated & np.isin(it, ["shipping", "billing"])] = "support_agent"
    esc_type[escalated & (it == "refill")] = "pharmacist"
    dose_side = escalated & np.isin(it, ["dosing_question", "side_effects"])
    esc_type[dose_side] = np.where(u[dose_side] < 0.7, "clinician", "pharmacist")
    # Symptom concerns from sicker patients are more often urgent.
    symptom = escalated & (it == "symptom_concern")
    p_urgent = 0.2 + 0.6 * c["risk"].to_numpy()[symptom]
    esc_type[symptom] = np.where(u[symptom] < p_urgent, "urgent_clinical", "clinician")
    c["escalation_type"] = esc_type

    # Resolution: depends on who handled it.
    res_logit = np.where(
        escalated,
        RES_HUMAN_INTERCEPT,
        RES_AI_INTERCEPT + RES_AI_CONFIDENCE_COEF * (confidence - 0.6),
    ) + RES_RISK_COEF * c_risk
    resolved = rng.random(n) < logistic(res_logit)
    c["resolved"] = resolved

    # CSAT: a latent "how did that feel" score, rounded to 1-5, observed only
    # if the patient answered the survey.
    latent = (
        CSAT_BASE
        + CSAT_RESOLVED_EFFECT * resolved
        + CSAT_ESCALATED_EFFECT * escalated
        + CSAT_CONFIDENCE_EFFECT * (confidence - 0.75)
        + CSAT_RISK_EFFECT * c_risk
        + rng.normal(0, CSAT_NOISE_SD, n)
    )
    csat = np.clip(np.round(latent), 1, 5)
    p_respond = logistic(CSAT_RESPONSE_INTERCEPT + CSAT_RESPONSE_EXTREMITY * np.abs(latent - CSAT_BASE))
    responded = rng.random(n) < p_respond
    c["csat"] = pd.array(np.where(responded, csat, np.nan), dtype="Int64")

    c = c.sort_values("started_at").reset_index(drop=True)
    c.insert(0, "conversation_id", np.arange(1, n + 1))
    return c[[
        "conversation_id", "patient_id", "started_at", "intent", "confidence",
        "escalated", "escalation_type", "resolved", "csat",
    ]]


def print_checks(patients, conversations):
    """Quick sanity checks: did the self-selection actually show up?"""
    per_patient = conversations.groupby("patient_id").size()
    p = patients.assign(
        n_convos=patients["patient_id"].map(per_patient).fillna(0).astype(int),
        risk_quintile=pd.qcut(patients["baseline_risk"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]),
    )
    p["used_chat"] = p["n_convos"] > 0
    print(f"patients: {len(patients):,}   conversations: {len(conversations):,}\n")
    print("Self-selection check (chat use by baseline risk quintile):")
    print(p.groupby("risk_quintile", observed=True).agg(
        mean_risk=("baseline_risk", "mean"),
        pct_used_chat=("used_chat", "mean"),
        convos_per_patient=("n_convos", "mean"),
    ).round(3).to_string())
    print("\nOutcome rates by intent:")
    print(conversations.groupby("intent").agg(
        share=("conversation_id", lambda s: len(s) / len(conversations)),
        mean_confidence=("confidence", "mean"),
        escalated=("escalated", "mean"),
        resolved=("resolved", "mean"),
        csat_mean=("csat", "mean"),
        csat_response=("csat", lambda s: s.notna().mean()),
    ).round(3).to_string())


def main():
    rng = np.random.default_rng(SEED)
    patients = generate_patients(rng)
    conversations = generate_conversations(patients, rng)

    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    patients.to_csv(out_dir / "patients.csv", index=False)
    conversations.to_csv(out_dir / "conversations.csv", index=False)
    print(f"Wrote {out_dir / 'patients.csv'} and {out_dir / 'conversations.csv'}\n")
    print_checks(patients, conversations)


if __name__ == "__main__":
    main()

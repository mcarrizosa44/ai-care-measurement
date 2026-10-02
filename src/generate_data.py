"""
Generate synthetic data for a telehealth company with an AI care chat.

Run:
    python src/generate_data.py

Outputs (in data/raw/):
    patients.csv              one row per patient
    subscriptions.csv         one row per patient: start and cancel date
    ai_launches.csv           when the AI chat launched for each program
    conversations.csv         one row per AI chat conversation
    support_tickets.csv       one row per human-handled contact (direct, or escalated from AI)
    qa_reviews.csv            human QA labels for a random 5% of AI conversations
    experiment_exposures.csv  GrowthBook-style A/B test assignments (model v1 vs v2)
    answer_key.json           the TRUE effects baked into the data. Don't open it until
                              you've done an analysis, then check whether you recovered them.

The story the data tells
------------------------
1. Patients sign up for a program (weight, hair, BP) and pay monthly, quarterly or annually.
2. The AI chat launches at a different date for each program (staggered rollout).
3. Each patient has questions over time ("contacts"). Before launch, every contact
   goes to a human support ticket. After launch, patients who are "chat adopters"
   send their contacts to the AI chat instead. The AI either handles it or escalates
   to a human, which also creates a ticket.
4. The AI chat truly reduces churn for adopters, but a naive comparison of chat users vs
   non-users gets the size wrong. Biases pull in both directions:
     - sicker patients adopt chat more AND churn more (makes chat look worse)
     - patients who stay longer have more chances to chat (makes chat look better;
       this is reverse causation / immortal-time bias)
     - engaged patients adopt chat more AND churn less (makes chat look better)
   In this data the naive comparison overstates the benefit by a lot.
5. A hidden "engagement" trait (never saved to any CSV) also makes people both more
   likely to adopt chat and less likely to churn. Methods that only adjust for observed
   columns can't fully remove it. Within-patient and difference-in-differences designs can.
6. In late 2025 an A/B test compares model v1 to v2. v2 is more confident and better at
   logistics questions, but overconfident on clinical questions, so it escalates
   symptom concerns less often. That's a safety guardrail problem to catch.
7. A few realistic data-quality bugs are injected for dbt tests to find.

Every number below is an assumption, collected at the top as a named constant so you can
point to any one of them and explain what it does and why.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# Settings
# ===========================================================================

# WHY a fixed seed: anyone who runs the script gets the exact same data.
SEED = 42
N_PATIENTS = 20_000
DAYS_PER_MONTH = 30.44

# Signups run for 18 months; data ends 6 months after the last signup.
SIGNUP_START = pd.Timestamp("2024-01-01")
SIGNUP_END = pd.Timestamp("2025-06-30")
OBSERVATION_END = pd.Timestamp("2025-12-31")

# --- Patients ---------------------------------------------------------------
# Weight management is the biggest line; BP is the smallest and most clinical.
PROGRAM_SHARES = {"weight": 0.55, "hair": 0.30, "bp": 0.15}
PLAN_SHARES = {"monthly": 0.60, "quarterly": 0.25, "annual": 0.15}
CHANNEL_SHARES = {"paid_social": 0.45, "search": 0.30, "referral": 0.15, "organic": 0.10}

# Age differs by program (hair loss skews young, hypertension skews older).
AGE_MEAN_SD = {"weight": (42, 11), "hair": (33, 8), "bp": (55, 10)}

# baseline_risk: 0-1 clinical complexity score at signup.
# WHY Beta: bounded 0-1, right-skewed (most patients low risk, long tail).
#   weight Beta(2, 5) mean ~0.29 | hair Beta(1.5, 8) mean ~0.16 | bp Beta(3, 4) mean ~0.43
RISK_BETA = {"weight": (2.0, 5.0), "hair": (1.5, 8.0), "bp": (3.0, 4.0)}
# Older patients within a program are a bit riskier, so age is a (mild) confounder too.
RISK_AGE_COEF = 0.004  # +0.04 risk per 10 years above the program's mean age

# All risk effects use (risk - RISK_CENTER), so intercepts describe a typical patient.
RISK_CENTER = 0.3

# --- AI chat launch (staggered rollout -> difference-in-differences) ---------
# WHY staggered: launching by program at different dates gives "not yet treated"
# programs that act as controls. That's the setup for difference-in-differences.
AI_LAUNCH_DATES = {
    "weight": pd.Timestamp("2024-04-01"),
    "hair": pd.Timestamp("2024-10-01"),
    "bp": pd.Timestamp("2025-03-01"),
}

# --- Chat adoption (the self-selection) -------------------------------------
# An "adopter" sends their questions to the AI chat once it's available.
# Risk and the hidden engagement trait both raise adoption.
ADOPT_INTERCEPT = -0.3   # ~43% adoption at risk 0.3, average engagement
ADOPT_RISK_COEF = 3.0    # ~29% at risk 0.1, ~70% at risk 0.7
ADOPT_ENGAGEMENT_COEF = 0.5

# --- Churn (discrete-time monthly hazard) -----------------------------------
# WHY monthly: subscriptions renew in billing periods, so "did they cancel at the end of
# this period?" is the natural unit. Each 30-day period is one logistic draw.
CHURN_BASE_LOGIT = {"monthly": -2.6, "quarterly": -3.0, "annual": -3.6}  # ~7%, ~5%, ~3% per month
CHURN_TENURE_COEF = -0.3          # churn is highest early, then falls (times log of period number)
CHURN_RISK_COEF = 1.0             # sicker patients churn more
CHURN_ENGAGEMENT_COEF = -0.3      # engaged patients churn less (unobserved!)
CHURN_PROGRAM_EFFECT = {"weight": 0.0, "hair": -0.1, "bp": -0.3}  # chronic meds are stickier
CHURN_CHANNEL_EFFECT = {"paid_social": 0.2, "search": 0.0, "referral": -0.2, "organic": -0.1}
# THE TRUE CAUSAL EFFECT: once chat is available, adopters' monthly churn log-odds drop by 0.25.
TRUE_CHAT_EFFECT_ON_CHURN_LOGODDS = -0.25

# --- Contacts (questions patients have) -------------------------------------
# Contact rate per month at signup for a typical patient. Rises with risk.
CONTACTS_PER_MONTH_AT_SIGNUP = 1.0
CONTACT_RISK_COEF = 1.5
# WHY decay with tenure: patients ask most questions during onboarding and first refills.
# Rate multiplier = floor + (1 - floor) * exp(-tenure_days / decay_days).
CONTACT_TENURE_FLOOR = 0.5
CONTACT_TENURE_DECAY_DAYS = 60
# WHY a gamma "chattiness" multiplier (mean 1): some people just ask more, regardless of risk.
CHATTINESS_GAMMA_SHAPE = 2.0
# WHY induced demand: when asking is easy, people ask more. So AI deflection is NOT 1:1;
# not every AI conversation would have been a human ticket.
AI_INDUCED_DEMAND = 1.3

# --- Intent -----------------------------------------------------------------
INTENTS = ["refill", "shipping", "billing", "dosing_question", "side_effects", "symptom_concern"]
CLINICAL_INTENTS = {"dosing_question", "side_effects", "symptom_concern"}
INTENT_BASE_PROBS = {
    #          refill ship  bill  dose  side  symptom
    "weight": [0.22, 0.18, 0.15, 0.17, 0.18, 0.10],
    "hair":   [0.25, 0.25, 0.20, 0.10, 0.15, 0.05],
    "bp":     [0.25, 0.12, 0.12, 0.18, 0.13, 0.20],
}
# Higher risk shifts the mix toward clinical topics (multinomial logit).
INTENT_RISK_COEF = {
    "refill": 0.0, "shipping": 0.0, "billing": 0.0,
    "dosing_question": 1.0, "side_effects": 1.5, "symptom_concern": 2.5,
}

# --- AI model behavior ------------------------------------------------------
# confidence: the model's self-reported confidence. Easy questions -> high confidence.
CONFIDENCE_MEAN = {
    "refill": 0.88, "shipping": 0.90, "billing": 0.85,
    "dosing_question": 0.75, "side_effects": 0.70, "symptom_concern": 0.60,
}
CONFIDENCE_CONCENTRATION = 12.0
# Intent classification: P(predicted intent is right) = confidence + 0.10 (capped at 0.99).
INTENT_ACCURACY_BOOST = 0.10
# Calibration gap: P(answer is actually correct) = confidence - gap.
# WHY: real models are overconfident. QA reviews let you measure this.
CALIBRATION_GAP = {
    ("v1", False): 0.05, ("v1", True): 0.05,   # (model, is_clinical)
    ("v2", False): 0.02, ("v2", True): 0.12,   # v2 is better on logistics, overconfident on clinical
}
P_CORRECT_IF_MISCLASSIFIED = 0.10

# --- A/B test: model v1 vs v2 (GrowthBook-style) -----------------------------
EXPERIMENT_ID = "ai-model-v2"
EXPERIMENT_START = pd.Timestamp("2025-08-01")
EXPERIMENT_END = pd.Timestamp("2025-11-01")   # exclusive
V2_CONFIDENCE_LIFT = 0.05

# --- Escalation -------------------------------------------------------------
# WHY a hard threshold: AI care products usually have a rule like "below X confidence,
# always hand off to a human". This also enables a regression discontinuity design.
ESCALATION_CONFIDENCE_THRESHOLD = 0.50
ESC_INTERCEPT = -1.8
ESC_CONFIDENCE_COEF = 6.0      # lower confidence -> more escalation
ESC_RISK_COEF = 1.5            # sicker patients get escalated more
ESC_WRONG_ANSWER_COEF = 1.0    # a wrong answer makes the patient ask for a human
ESC_INTENT_BUMP = {
    "refill": 0.0, "shipping": 0.0, "billing": 0.0,
    "dosing_question": 0.5, "side_effects": 0.8, "symptom_concern": 1.5,
}
# Which human queue handles each intent. Symptom concerns can be urgent.
QUEUE_BY_INTENT = {
    "shipping": "support_agent", "billing": "support_agent", "refill": "pharmacist",
    "dosing_question": "clinician", "side_effects": "clinician", "symptom_concern": "clinician",
}

# --- Resolution -------------------------------------------------------------
RES_AI_INTERCEPT = -1.0
RES_AI_CORRECT_COEF = 2.5      # correct answer: ~82% resolved; wrong: ~27%
RES_HUMAN_INTERCEPT = 1.2      # humans resolve ~77% of their contacts
RES_RISK_COEF = -0.8

# --- Human handle time (for cost and efficiency analysis) --------------------
# WHY lognormal: handle times are positive and right-skewed (a few calls run long).
HANDLE_MINUTES_MEDIAN = {
    "shipping": 6, "billing": 7, "refill": 8,
    "dosing_question": 12, "side_effects": 14, "symptom_concern": 18,
}
HANDLE_MINUTES_SIGMA = 0.5
URGENT_HANDLE_MULT = 1.4
# The AI passes a conversation summary on handoff, so agents work a bit faster.
AI_HANDOFF_HANDLE_MULT = 0.9

# --- CSAT (AI conversations only) --------------------------------------------
# WHY model nonresponse: people with extreme experiences answer surveys more often, so
# missing CSAT is NOT random. Averaging the csat column is biased.
CSAT_BASE = 3.2
CSAT_RESOLVED_EFFECT = 1.2
CSAT_ESCALATED_EFFECT = -0.4
CSAT_RISK_EFFECT = -0.5
CSAT_NOISE_SD = 0.9
CSAT_RESPONSE_INTERCEPT = -0.8
CSAT_RESPONSE_EXTREMITY = 0.6

# --- QA sample ---------------------------------------------------------------
QA_SAMPLE_RATE = 0.05

# --- Injected data-quality issues (for dbt tests to catch) -------------------
INJECT_DATA_QUALITY_ISSUES = True
DQ_DUPLICATE_RATE = 0.005                      # event re-sent: exact duplicate conversation rows
DQ_MISSING_ESC_TYPE_WINDOW = ("2025-05-12", "2025-05-22")  # logging bug after a schema change
DQ_CSAT_10PT_WINDOW = ("2024-11-01", "2024-11-15")         # survey vendor test used a 2-10 scale


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# ===========================================================================
# Patients
# ===========================================================================

def generate_patients(rng):
    n = N_PATIENTS

    # WHY a rising ramp: the business is growing, so later days get more signups.
    days = pd.date_range(SIGNUP_START, SIGNUP_END, freq="D")
    weights = np.linspace(1.0, 2.0, len(days))
    signup_date = rng.choice(days, size=n, p=weights / weights.sum())

    program = rng.choice(list(PROGRAM_SHARES), size=n, p=list(PROGRAM_SHARES.values()))
    plan = rng.choice(list(PLAN_SHARES), size=n, p=list(PLAN_SHARES.values()))
    channel = rng.choice(list(CHANNEL_SHARES), size=n, p=list(CHANNEL_SHARES.values()))

    age = np.empty(n)
    risk = np.empty(n)
    for prog in PROGRAM_SHARES:
        mask = program == prog
        mean_age, sd_age = AGE_MEAN_SD[prog]
        age[mask] = np.clip(rng.normal(mean_age, sd_age, mask.sum()), 18, 80)
        a, b = RISK_BETA[prog]
        risk[mask] = rng.beta(a, b, mask.sum()) + RISK_AGE_COEF * (age[mask] - mean_age)
    risk = np.clip(risk, 0.001, 0.999)

    return pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "signup_date": pd.to_datetime(signup_date).normalize(),
        "program": program,
        "plan": plan,
        "acquisition_channel": channel,
        "age": age.round().astype(int),
        "baseline_risk": risk.round(3),
    })


# ===========================================================================
# Churn
# ===========================================================================

def simulate_churn(patients, adopter, engagement, rng):
    """Walk through 30-day billing periods; at the end of each, an active patient may cancel.

    Returns cancel dates (NaT = still subscribed at OBSERVATION_END) and the true average
    effect of chat on monthly churn probability, for the answer key.
    """
    n = len(patients)
    signup = patients["signup_date"].to_numpy()
    launch = patients["program"].map(AI_LAUNCH_DATES).to_numpy()
    centered_risk = patients["baseline_risk"].to_numpy() - RISK_CENTER

    # The part of the churn log-odds that doesn't change over time.
    fixed_logit = (
        patients["plan"].map(CHURN_BASE_LOGIT).to_numpy()
        + CHURN_RISK_COEF * centered_risk
        + CHURN_ENGAGEMENT_COEF * engagement
        + patients["program"].map(CHURN_PROGRAM_EFFECT).to_numpy()
        + patients["acquisition_channel"].map(CHURN_CHANNEL_EFFECT).to_numpy()
    )

    cancel = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    active = np.ones(n, dtype=bool)
    effect_sum, effect_count = 0.0, 0
    period = 1
    while active.any():
        period_end = signup + np.timedelta64(30 * period, "D")
        period_start = period_end - np.timedelta64(30, "D")
        # If the period ends after the data does, we never see whether they cancel (censored).
        active &= period_end <= np.datetime64(OBSERVATION_END)

        chat_on = adopter & (period_start >= launch)
        logit_no_chat = fixed_logit + CHURN_TENURE_COEF * np.log(period)
        logit = logit_no_chat + TRUE_CHAT_EFFECT_ON_CHURN_LOGODDS * chat_on
        churned = active & (rng.random(n) < logistic(logit))

        # Answer key: how much did chat change churn probability, in the periods where it applied?
        treated = active & chat_on
        effect_sum += (logistic(logit[treated]) - logistic(logit_no_chat[treated])).sum()
        effect_count += treated.sum()

        cancel[churned] = period_end[churned]
        active &= ~churned
        period += 1

    return cancel, effect_sum / effect_count


# ===========================================================================
# Contacts -> AI conversations or human tickets
# ===========================================================================

def draw_intents(program, centered_risk, rng):
    """Multinomial logit: log(base prob for program) + risk shift, then softmax and draw."""
    risk_coefs = np.array([INTENT_RISK_COEF[i] for i in INTENTS])
    intent = np.empty(len(program), dtype=object)
    for prog, probs in INTENT_BASE_PROBS.items():
        mask = program == prog
        p = np.exp(np.log(probs) + np.outer(centered_risk[mask], risk_coefs))
        p /= p.sum(axis=1, keepdims=True)
        u = rng.random(mask.sum())[:, None]
        intent[mask] = np.array(INTENTS)[(u > p.cumsum(axis=1)).sum(axis=1)]
    return intent


def draw_handle_minutes(intent, urgent, from_ai, rng):
    median = pd.Series(intent).map(HANDLE_MINUTES_MEDIAN).to_numpy(dtype=float)
    median *= np.where(urgent, URGENT_HANDLE_MULT, 1.0)
    median *= np.where(from_ai, AI_HANDOFF_HANDLE_MULT, 1.0)
    return np.round(median * np.exp(rng.normal(0, HANDLE_MINUTES_SIGMA, len(intent))), 1)


def generate_contacts(patients, cancel, adopter, rng):
    """Every question a patient has, with a timestamp and a route (AI or human).

    WHY "thinning": the contact rate changes over time (tenure decay, and induced demand
    once AI is on). We generate candidate times at the maximum possible rate, then keep
    each one with probability (actual rate / max rate). That gives exactly the right
    time-varying rate.
    """
    signup = patients["signup_date"].to_numpy()
    end = np.where(pd.isna(cancel), np.datetime64(OBSERVATION_END), cancel)
    window_days = (end - signup) / np.timedelta64(1, "D")
    centered_risk = patients["baseline_risk"].to_numpy() - RISK_CENTER

    chattiness = rng.gamma(CHATTINESS_GAMMA_SHAPE, 1 / CHATTINESS_GAMMA_SHAPE, len(patients))
    base_rate = CONTACTS_PER_MONTH_AT_SIGNUP * np.exp(CONTACT_RISK_COEF * centered_risk) * chattiness
    max_rate = base_rate * AI_INDUCED_DEMAND

    n_candidates = rng.poisson(max_rate * window_days / DAYS_PER_MONTH)
    idx = np.repeat(np.arange(len(patients)), n_candidates)
    tenure_days = rng.random(len(idx)) * window_days[idx]
    ts = signup[idx] + (tenure_days * 86400).astype("timedelta64[s]")

    launch = patients["program"].map(AI_LAUNCH_DATES).to_numpy()
    ai_on = adopter[idx] & (ts >= launch[idx])
    tenure_mult = CONTACT_TENURE_FLOOR + (1 - CONTACT_TENURE_FLOOR) * np.exp(-tenure_days / CONTACT_TENURE_DECAY_DAYS)
    keep_prob = tenure_mult * np.where(ai_on, AI_INDUCED_DEMAND, 1.0) / AI_INDUCED_DEMAND
    keep = rng.random(len(idx)) < keep_prob

    idx, ts, ai_on = idx[keep], ts[keep], ai_on[keep]
    program = patients["program"].to_numpy()[idx]
    risk = patients["baseline_risk"].to_numpy()[idx]
    return pd.DataFrame({
        "patient_id": patients["patient_id"].to_numpy()[idx],
        "ts": pd.to_datetime(ts).floor("min"),
        "program": program,
        "risk": risk,
        "to_ai": ai_on,
        "true_intent": draw_intents(program, risk - RISK_CENTER, rng),
    }).sort_values("ts", ignore_index=True)


def assign_experiment(ai_contacts, rng):
    """GrowthBook-style: a patient is assigned when first exposed (their first AI chat in
    the experiment window), 50/50, and keeps that variation for the whole test."""
    in_window = (ai_contacts["ts"] >= EXPERIMENT_START) & (ai_contacts["ts"] < EXPERIMENT_END)
    first = ai_contacts[in_window].groupby("patient_id")["ts"].min()
    variation = rng.integers(0, 2, len(first))
    exposures = pd.DataFrame({
        "patient_id": first.index,
        "experiment_id": EXPERIMENT_ID,
        "variation_id": variation,
        "variation_name": np.where(variation == 1, "v2", "v1"),
        "first_exposure_at": first.to_numpy(),
    })
    v2_patients = set(exposures.loc[exposures["variation_id"] == 1, "patient_id"])
    model_version = np.where(in_window & ai_contacts["patient_id"].isin(v2_patients), "v2", "v1")
    return exposures, model_version


def simulate_ai_conversations(c, rng):
    """Given AI contacts (true intent, risk, model version), simulate what the model does."""
    n = len(c)
    c_risk = c["risk"].to_numpy() - RISK_CENTER
    true_intent = c["true_intent"].to_numpy()
    is_v2 = c["model_version"].to_numpy() == "v2"
    true_clinical = np.isin(true_intent, list(CLINICAL_INTENTS))

    # Confidence: Beta(mean*k, (1-mean)*k) has the given mean; k controls spread.
    mean_conf = pd.Series(true_intent).map(CONFIDENCE_MEAN).to_numpy() + V2_CONFIDENCE_LIFT * is_v2
    mean_conf = np.clip(mean_conf, 0.01, 0.97)
    k = CONFIDENCE_CONCENTRATION
    confidence = rng.beta(mean_conf * k, (1 - mean_conf) * k).round(3)

    # Intent classification: right most of the time, more often when confident.
    classified_right = rng.random(n) < np.minimum(confidence + INTENT_ACCURACY_BOOST, 0.99)
    other_intent = np.array(INTENTS)[rng.integers(0, len(INTENTS), n)]
    # If the random "other" happens to equal the true intent, it's a correct guess anyway.
    predicted = np.where(classified_right, true_intent, other_intent)
    classified_right = predicted == true_intent

    # Was the answer actually correct? (Only visible later, through QA reviews.)
    gap = np.array([CALIBRATION_GAP[("v2" if v else "v1", cl)] for v, cl in zip(is_v2, true_clinical)])
    p_correct = np.where(classified_right, np.clip(confidence - gap, 0, 1), P_CORRECT_IF_MISCLASSIFIED)
    answer_correct = rng.random(n) < p_correct

    # Escalation: forced below the threshold, a judgment call above it.
    # The model only knows what it PREDICTED the intent to be, so routing uses `predicted`.
    esc_logit = (
        ESC_INTERCEPT
        + ESC_CONFIDENCE_COEF * (0.75 - confidence)
        + ESC_RISK_COEF * c_risk
        + ESC_WRONG_ANSWER_COEF * ~answer_correct
        + pd.Series(predicted).map(ESC_INTENT_BUMP).to_numpy()
    )
    escalated = (confidence < ESCALATION_CONFIDENCE_THRESHOLD) | (rng.random(n) < logistic(esc_logit))

    # Escalation type = which human queue. Symptom concerns from sicker patients are often urgent.
    queue = pd.Series(predicted).map(QUEUE_BY_INTENT).to_numpy().astype(object)
    urgent = (predicted == "symptom_concern") & (rng.random(n) < 0.2 + 0.6 * c["risk"].to_numpy())
    queue[urgent] = "urgent_clinical"
    escalation_type = np.where(escalated, queue, "")

    # Resolution: a human resolves most handoffs; the AI resolves mostly when it's right.
    res_logit = np.where(
        escalated,
        RES_HUMAN_INTERCEPT,
        RES_AI_INTERCEPT + RES_AI_CORRECT_COEF * answer_correct,
    ) + RES_RISK_COEF * c_risk
    resolved = rng.random(n) < logistic(res_logit)

    # CSAT: latent satisfaction -> 1-5, observed only if the patient answers the survey.
    latent = (
        CSAT_BASE
        + CSAT_RESOLVED_EFFECT * resolved
        + CSAT_ESCALATED_EFFECT * escalated
        + CSAT_RISK_EFFECT * c_risk
        + rng.normal(0, CSAT_NOISE_SD, n)
    )
    responded = rng.random(n) < logistic(CSAT_RESPONSE_INTERCEPT + CSAT_RESPONSE_EXTREMITY * np.abs(latent - CSAT_BASE))
    csat = np.where(responded, np.clip(np.round(latent), 1, 5), np.nan)

    return c.assign(
        intent=predicted, confidence=confidence, escalated=escalated,
        escalation_type=escalation_type, urgent=urgent & escalated,
        resolved=resolved, csat=pd.array(csat, dtype="Int64"), answer_correct=answer_correct,
    )


def build_tickets(direct, conversations, rng):
    """Human-handled contacts: direct tickets (no AI) plus handoffs from the AI."""
    n_direct = len(direct)
    d_risk = direct["risk"].to_numpy() - RISK_CENTER
    d_intent = direct["true_intent"].to_numpy()
    d_queue = pd.Series(d_intent).map(QUEUE_BY_INTENT).to_numpy().astype(object)
    d_urgent = (d_intent == "symptom_concern") & (rng.random(n_direct) < 0.2 + 0.6 * direct["risk"].to_numpy())
    d_queue[d_urgent] = "urgent_clinical"
    direct_tickets = pd.DataFrame({
        "patient_id": direct["patient_id"].to_numpy(),
        "created_at": direct["ts"].to_numpy(),
        "source": "direct",
        "conversation_id": pd.array([pd.NA] * n_direct, dtype="Int64"),
        "intent": d_intent,
        "queue": d_queue,
        "handle_minutes": draw_handle_minutes(d_intent, d_urgent, False, rng),
        "resolved": rng.random(n_direct) < logistic(RES_HUMAN_INTERCEPT + RES_RISK_COEF * d_risk),
    })

    # Handoffs: the human agent tags the TRUE intent (they can read the conversation).
    esc = conversations[conversations["escalated"]]
    handoff_tickets = pd.DataFrame({
        "patient_id": esc["patient_id"].to_numpy(),
        "created_at": esc["started_at"].to_numpy() + np.timedelta64(5, "m"),
        "source": "ai_escalation",
        "conversation_id": pd.array(esc["conversation_id"].to_numpy(), dtype="Int64"),
        "intent": esc["true_intent"].to_numpy(),
        "queue": esc["escalation_type"].to_numpy(),
        "handle_minutes": draw_handle_minutes(esc["true_intent"].to_numpy(), esc["urgent"].to_numpy(), True, rng),
        "resolved": esc["resolved"].to_numpy(),
    })

    tickets = pd.concat([direct_tickets, handoff_tickets], ignore_index=True)
    tickets = tickets.sort_values("created_at", ignore_index=True)
    tickets.insert(0, "ticket_id", np.arange(1, len(tickets) + 1))
    return tickets


def inject_data_quality_issues(conversations, rng):
    c = conversations.copy()
    # 1. Logging bug: escalation_type not recorded for 10 days after a schema change.
    start, end = map(pd.Timestamp, DQ_MISSING_ESC_TYPE_WINDOW)
    in_window = (c["started_at"] >= start) & (c["started_at"] < end)
    c.loc[in_window, "escalation_type"] = ""
    # 2. Survey vendor test: CSAT recorded on a 2-10 scale for two weeks.
    start, end = map(pd.Timestamp, DQ_CSAT_10PT_WINDOW)
    in_window = (c["started_at"] >= start) & (c["started_at"] < end)
    c.loc[in_window, "csat"] = c.loc[in_window, "csat"] * 2
    # 3. Events re-sent: a few exact duplicate rows.
    dupes = c.sample(frac=DQ_DUPLICATE_RATE, random_state=int(rng.integers(1e9)))
    return pd.concat([c, dupes]).sort_values(["started_at", "conversation_id"], ignore_index=True)


# ===========================================================================
# Main
# ===========================================================================

def main():
    rng = np.random.default_rng(SEED)

    patients = generate_patients(rng)

    # Hidden engagement trait: affects adoption and churn, never saved. (Unobserved confounder.)
    engagement = rng.normal(0, 1, len(patients))
    centered_risk = patients["baseline_risk"].to_numpy() - RISK_CENTER
    p_adopt = logistic(ADOPT_INTERCEPT + ADOPT_RISK_COEF * centered_risk + ADOPT_ENGAGEMENT_COEF * engagement)
    adopter = rng.random(len(patients)) < p_adopt

    cancel, true_effect_pp = simulate_churn(patients, adopter, engagement, rng)
    subscriptions = pd.DataFrame({
        "patient_id": patients["patient_id"],
        "start_date": patients["signup_date"],
        "cancel_date": pd.to_datetime(cancel),
    })

    contacts = generate_contacts(patients, cancel, adopter, rng)
    ai = contacts[contacts["to_ai"]].reset_index(drop=True)
    exposures, ai["model_version"] = assign_experiment(ai, rng)

    conversations = simulate_ai_conversations(ai, rng)
    conversations = conversations.rename(columns={"ts": "started_at"})
    conversations.insert(0, "conversation_id", np.arange(1, len(conversations) + 1))

    tickets = build_tickets(contacts[~contacts["to_ai"]], conversations, rng)

    qa = conversations.sample(frac=QA_SAMPLE_RATE, random_state=int(rng.integers(1e9)))
    qa_reviews = pd.DataFrame({
        "conversation_id": qa["conversation_id"].to_numpy(),
        "reviewed_at": (qa["started_at"] + pd.to_timedelta(rng.integers(1, 15, len(qa)), unit="D")).dt.normalize().to_numpy(),
        "true_intent": qa["true_intent"].to_numpy(),
        "answer_correct": qa["answer_correct"].to_numpy(),
    }).sort_values("conversation_id", ignore_index=True)

    conversations_out = conversations[[
        "conversation_id", "patient_id", "started_at", "model_version", "intent", "confidence",
        "escalated", "escalation_type", "resolved", "csat",
    ]]
    if INJECT_DATA_QUALITY_ISSUES:
        conversations_out = inject_data_quality_issues(conversations_out, rng)

    ai_launches = pd.DataFrame({"program": list(AI_LAUNCH_DATES), "launch_date": list(AI_LAUNCH_DATES.values())})

    answer_key = {
        "unobserved_confounder": "engagement: raises chat adoption (+0.5 log-odds per SD) and lowers churn (-0.3 log-odds per SD). Not in any CSV.",
        "chat_effect_on_churn": {
            "who": "chat adopters, in billing periods after their program's AI launch",
            "monthly_churn_log_odds": TRUE_CHAT_EFFECT_ON_CHURN_LOGODDS,
            "monthly_churn_odds_ratio": round(float(np.exp(TRUE_CHAT_EFFECT_ON_CHURN_LOGODDS)), 3),
            "avg_change_in_monthly_churn_prob_pp": round(float(true_effect_pp) * 100, 3),
        },
        "adoption_rate_by_program": {
            prog: round(float(adopter[patients["program"] == prog].mean()), 3) for prog in AI_LAUNCH_DATES
        },
        "ai_launch_dates": {k: str(v.date()) for k, v in AI_LAUNCH_DATES.items()},
        "ai_induced_demand_multiplier": AI_INDUCED_DEMAND,
        "ai_handoff_handle_time_multiplier": AI_HANDOFF_HANDLE_MULT,
        "escalation_confidence_threshold": ESCALATION_CONFIDENCE_THRESHOLD,
        "experiment": {
            "id": EXPERIMENT_ID,
            "window": [str(EXPERIMENT_START.date()), str(EXPERIMENT_END.date()) + " (exclusive)"],
            "v2_confidence_lift": V2_CONFIDENCE_LIFT,
            "calibration_gap": {f"{m}_{'clinical' if cl else 'nonclinical'}": g for (m, cl), g in CALIBRATION_GAP.items()},
            "expected_finding": "v2 resolves more logistics questions, but is overconfident on clinical ones and escalates symptom concerns less: a safety guardrail failure.",
        },
        "data_quality_issues": [
            f"~{DQ_DUPLICATE_RATE:.1%} of conversation rows are exact duplicates (same conversation_id)",
            f"escalation_type is blank for escalated conversations from {DQ_MISSING_ESC_TYPE_WINDOW[0]} to {DQ_MISSING_ESC_TYPE_WINDOW[1]} (exclusive)",
            f"csat is on a 2-10 scale from {DQ_CSAT_10PT_WINDOW[0]} to {DQ_CSAT_10PT_WINDOW[1]} (exclusive)",
        ],
    }

    out = Path(__file__).resolve().parent.parent / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    patients.assign(signup_date=patients["signup_date"].dt.date).to_csv(out / "patients.csv", index=False)
    subscriptions.assign(
        start_date=subscriptions["start_date"].dt.date, cancel_date=subscriptions["cancel_date"].dt.date
    ).to_csv(out / "subscriptions.csv", index=False)
    ai_launches.assign(launch_date=ai_launches["launch_date"].dt.date).to_csv(out / "ai_launches.csv", index=False)
    conversations_out.to_csv(out / "conversations.csv", index=False)
    tickets.to_csv(out / "support_tickets.csv", index=False)
    qa_reviews.to_csv(out / "qa_reviews.csv", index=False)
    exposures.to_csv(out / "experiment_exposures.csv", index=False)
    (out / "answer_key.json").write_text(json.dumps(answer_key, indent=2))

    print_checks(patients, subscriptions, conversations_out, tickets, qa_reviews, exposures)


def print_checks(patients, subscriptions, conversations, tickets, qa_reviews, exposures):
    print("Rows written:")
    for name, df in [("patients", patients), ("subscriptions", subscriptions), ("conversations", conversations),
                     ("support_tickets", tickets), ("qa_reviews", qa_reviews), ("experiment_exposures", exposures)]:
        print(f"  {name:22s}{len(df):>8,}")

    users = set(conversations["patient_id"])
    p = patients.assign(
        used_ai_chat=patients["patient_id"].isin(users),
        cancelled=subscriptions["cancel_date"].notna().to_numpy(),
        risk_quintile=pd.qcut(patients["baseline_risk"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]),
    )
    print("\nSelf-selection: AI chat use by risk quintile")
    print(p.groupby("risk_quintile", observed=True)[["baseline_risk", "used_ai_chat"]].mean().round(3).to_string())
    print("\nNaive comparison (biased on purpose): share who cancelled")
    print(p.groupby("used_ai_chat")["cancelled"].mean().round(3).to_string())
    print("\nExperiment split:", exposures["variation_name"].value_counts().to_dict())


if __name__ == "__main__":
    main()

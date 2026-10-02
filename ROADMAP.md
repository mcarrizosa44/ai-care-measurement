# Roadmap: Staff Analyst, AI

Each module covers part of the job description. Each one produces something you can show and a story you can tell in an interview. Do them in order, because later modules use earlier ones.

Check your work against `data/raw/answer_key.json` only **after** each analysis. The point is to find out whether your method recovered the truth.

---

## 0. Data and the business model ✅ (generator built)

**JD:** telehealth / subscription healthcare; multi-source data environment.

- Read `src/generate_data.py` until you can explain every line.
- Draw the data model: patients → subscriptions; contacts → AI conversations or tickets; conversations → QA reviews; exposures.
- **Be able to say:** "Here's how I'd expect an AI care chat to change patient behavior, and here are the three biases that make the naive number wrong."

## 1. dbt pipeline and data quality

**JD:** dbt models and warehouse tables; data integrity across event schemas; partnering with Analytics Engineering.

- Set up a dbt-duckdb project (DuckDB stands in for BigQuery).
- `staging/`: one clean model per raw CSV (types, names, deduplication).
- `marts/`: `fct_ai_conversations`, `fct_support_tickets`, `dim_patients`, `patient_months` (one row per patient per month, with chat use and churn: the panel for later modules).
- dbt tests that **catch the three injected bugs**: duplicate IDs, blank escalation_type when escalated, CSAT out of range. Write up how you'd fix each one upstream.
- Write a **tracking requirements spec**: the events and fields a new AI feature must log before launch.
- **Learn:** BigQuery SQL differences (`DATE_TRUNC`, `SAFE_DIVIDE`, `QUALIFY`, partitioning and clustering).

## 2. Metrics framework

**JD:** define, validate and own metrics frameworks; translate model signals into product and business metrics.

- A metric tree: north star (resolved contacts without a human) → containment, resolution, escalation by type, CSAT → cost per resolved contact, churn.
- Definitions doc: formula, grain, owner, known pitfalls. Examples: containment that counts abandoned chats as "contained"; CSAT nonresponse bias.
- Guardrails: urgent-clinical escalation rate, symptom-concern escalation rate, QA answer accuracy on clinical intents.
- **Be able to say:** why "containment" alone is a dangerous AI metric in healthcare.

## 3. Model performance → business metrics

**JD:** partner with ML teams; confidence, escalation type, resolution outcome; intent classification, hallucination and quality monitoring.

- From `qa_reviews`: intent-classification accuracy and a confusion matrix; answer accuracy by intent.
- **Calibration curve:** does 0.8 confidence mean 80% correct? (It doesn't; measure the gap.)
- **Threshold tradeoff:** sweep the escalation threshold (0.4 → 0.7) and plot containment vs clinical safety vs human cost. That's a product decision you could hand to leadership.
- QA sample sizing: how many reviews a week do you need to detect a 5-point drop in accuracy?

## 4. A/B test: model v1 vs v2

**JD:** experimentation strategy (GrowthBook); test design, success metrics, sample size, guardrails, results interpretation.

- **Pre-registration doc**, written *before* looking at results: hypothesis, primary metric, secondary metrics, guardrails, MDE, power and sample size, duration, decision rule.
- **Checks:** sample ratio mismatch (chi-square on the split), pre-exposure balance.
- **Analysis:** randomized by patient, measured by conversation, so use cluster-robust standard errors or the delta method. Add CUPED if you have a pre-period.
- **Readout memo:** v2 wins on the headline but fails a clinical-safety guardrail. What do you recommend? (Ship for non-clinical intents only? Recalibrate first?)
- **Learn:** how GrowthBook works (exposure events, Bayesian vs frequentist engines, sequential testing).

## 5. Causal inference without randomization

**JD:** fixed effects, difference-in-differences, propensity score matching; AI impact on retention.

Question: **does using the AI chat reduce churn?** Run these in order and watch the estimate move:

1. Naive comparison (badly biased).
2. Regression adjustment for risk, plan, program, channel, age.
3. **Propensity score matching** on observed covariates. Check balance (standardized mean differences). It still can't fix the hidden engagement trait.
4. **Patient fixed effects** (pyfixest) on `patient_months`: compare each patient to themselves before and after chat became available.
5. **Staggered difference-in-differences** by program launch date. Event-study plot, pre-trend check, and why plain two-way FE can mislead with staggered timing (look up Callaway & Sant'Anna).
6. **Regression discontinuity** at the 0.5 confidence threshold: what does forced escalation do to resolution and CSAT?

Compare every estimate to the answer key, and write one paragraph on why each method lands where it does.

## 6. Retention and survival

**JD:** survival/retention modeling; subscription business; cohort analysis; lifelines.

- Cohort retention curves by signup month, plan and program.
- Kaplan-Meier curves (lifelines). Then show the **immortal-time bias** trap: "ever used chat" is defined using future information.
- Cox model with chat use as a **time-varying covariate** (`CoxTimeVaryingFitter`). Check the proportional-hazards assumption.

## 7. Commercial impact and sizing

**JD:** cost savings, handle-time efficiency, support deflection; churn addressability; sizing AI initiatives.

- **Deflection:** human tickets per active patient-month, before vs after launch (use DiD again). Explain why AI conversations ≠ tickets avoided (induced demand).
- **Handle time:** do AI handoffs take less human time than direct tickets of the same intent and queue?
- **Cost model:** a small, editable assumptions table (cost per agent minute, cost per AI conversation) → net savings, with ranges.
- **Churn addressability:** what share of churn sits among patients the AI could reach? Then, given the causal estimate from module 5, how many retained patients and how much revenue is that?

## 8. Deep dive: where the AI works and where it doesn't

**JD:** segmentation; identify where AI products are and aren't working; quantify opportunity; recommend actions.

- Segment by intent × risk × program: containment, resolution, CSAT, cost.
- Find the 2–3 biggest opportunities (e.g. "side-effect questions from high-risk weight patients: high volume, low resolution").
- End with **recommendations**, not charts.

## 9. Dashboard and executive communication

**JD:** Looker dashboards; self-service; explaining analyses to non-technical audiences and senior leadership.

- Build a dashboard on the dbt marts (Looker Studio is free and closest; or Evidence or Streamlit). Optionally write LookML views to show you know the semantic layer.
- **One-page exec memo** for the A/B result and one for the churn impact. Bottom line first, then uncertainty in plain language.
- Practice: explain DiD to a PM in 60 seconds.

## 10. Interview prep

- Map each module to a story: situation → what was ambiguous → what you did → the decision it changed.
- Tie in your conDati A/B story where it fits (module 4).
- Keep `ai-workflow.md` up to date. "How I use AI and where it's wrong" is a strong answer for an AI analytics role.

---

### Not covered here (be ready to talk about them)

- **BigQuery and Looker** specifically: the free tiers (BigQuery sandbox, Looker Studio) are enough to load the CSVs and practice.
- **Airflow / Databricks:** know what they do and where dbt fits (orchestration vs transformation vs compute).
- **LLM evaluation beyond QA labels:** LLM-as-judge, hallucination rubrics, offline eval sets vs online metrics.
- **Mentoring and influence:** have one story about raising a team's standards (definitions, code review, documentation).

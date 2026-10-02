# AI workflow log

How I'm using Claude on this project, what I verified myself, and what it got wrong.

## 2026-10-02: Repo setup and data generator

**Asked Claude to:** set up the repo and Python environment, and write the synthetic data generator from my starter prompt.

**What it produced:** `src/generate_data.py`, with each modeling choice explained as a `WHY` comment and every assumption collected as a named constant at the top.

**Things to verify / notes:**
- Claude didn't have my README template, so it wrote its own. Swap in mine or reconcile them.
- Claude added a few things my prompt didn't ask for. I need to decide whether to keep each one:
  - a hard escalation rule at confidence < 0.5 (to support a regression discontinuity design)
  - non-random CSAT nonresponse
  - exposure-time scaling (early signups have more time to chat)
- [ ] Read the generator until I can explain every line.
- [ ] Re-derive one or two numbers by hand, e.g. P(adopt chat) at risk 0.7 ≈ 70%.

**What Claude got wrong:**
- _(fill in as I find things)_

## 2026-10-02: Generator extended to cover the job description

**Asked Claude to:** cover everything the Staff Analyst, AI role expects.

**What it produced:** an extended generator (subscriptions and churn, staggered launch, support tickets with handle time, QA reviews, an A/B test, injected data bugs, an answer key) and `ROADMAP.md` mapping each JD bullet to a module.

**What Claude got wrong:**
- Its first docstring said the naive comparison makes chat users "look worse". Running the data showed the opposite: chat users cancel less (25% vs 40%), mostly because longer-tenured and more engaged patients have more chances to chat. It fixed the docstring. **Lesson:** check the direction of a bias against the data, not just the story.
- Its code produced NumPy deprecation warnings on Python 3.14. Fixed.

**To verify myself:**
- [ ] Is a 0.25 log-odds churn effect realistic, or too big? Why might it be?
- [ ] Is it realistic that adopters send *all* their contacts to chat? (No. Think about what this simplification hides.)

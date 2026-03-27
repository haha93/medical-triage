---
title: Medical Triage OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
---

# OpenEnv Medical Triage Environment

**Tags:** `openenv` · `medical` · `triage` · `healthcare`

An OpenEnv-compliant AI agent environment that simulates emergency room triage decisions. Agents receive structured patient cases — symptoms, vitals, medical history, demographics — and must assign urgency levels, route patients to the correct department, and recommend initial clinical actions. A multi-dimensional grader scores decisions against ground-truth triage protocols.

## Observation Space

Each observation represents a patient case with the following fields:

| Field | Type | Description |
|---|---|---|
| `patient_id` | `str` | Unique patient identifier |
| `symptoms` | `list[str]` | Presenting symptoms |
| `vitals.heart_rate` | `int` | Heart rate in bpm (30–250) |
| `vitals.blood_pressure` | `str` | Blood pressure as "systolic/diastolic" |
| `vitals.temperature` | `float` | Body temperature in °C (35.0–42.0) |
| `vitals.oxygen_saturation` | `int` | SpO2 percentage (50–100) |
| `vitals.respiratory_rate` | `int` | Breaths per minute (5–60) |
| `medical_history` | `list[str]` | Relevant past conditions |
| `age` | `int` | Patient age |
| `gender` | `str` | Patient gender |
| `chief_complaint` | `str` | Primary reason for visit |
| `time_of_arrival` | `str` | Arrival timestamp |
| `current_medications` | `list[str]` | Active medications |

## Action Space

Agents submit a triage decision with three components:

| Field | Type | Valid Values |
|---|---|---|
| `urgency_level` | `int` | 1 (life-threatening) to 5 (non-emergency) |
| `department` | `str` | `emergency`, `cardiology`, `orthopedics`, `neurology`, `general_medicine`, `pediatrics`, `pulmonology`, `gastroenterology`, `dermatology`, `psychiatry` |
| `initial_actions` | `list[str]` | `start_iv`, `order_ecg`, `administer_oxygen`, `pain_management`, `order_blood_work`, `order_ct_scan`, `order_xray`, `immobilize_limb`, `monitor_vitals`, `administer_epinephrine`, `order_mri`, `fluid_resuscitation`, `apply_tourniquet`, `administer_nitroglycerin`, `intubation_prep` |

## Reward Function

The grader computes a weighted score across four dimensions, clamped to [0.0, 1.0]:

| Component | Weight | Scoring |
|---|---|---|
| Urgency accuracy | 40% | 1.0 exact match · 0.5 off-by-one · 0.0 off-by-two+ |
| Department routing | 30% | 1.0 correct · 0.0 incorrect |
| Initial actions | 20% | Ratio of correct actions to ground-truth total (extra actions ignored) |
| Safety penalty | -10% | Applied when a critical patient (urgency 1–2) is under-triaged to level 4–5 |

**Formula:** `total = (urgency × 0.4) + (department × 0.3) + (actions × 0.2) + safety_penalty`

## Tasks

| Task | Difficulty | Cases | Description |
|---|---|---|---|
| `easy` | Easy | 10 | Single clear symptom, obvious department routing, unambiguous urgency |
| `medium` | Medium | 10 | Multiple symptoms, some diagnostic ambiguity, requires priority reasoning |
| `hard` | Hard | 10 | Complex multi-condition presentations, conflicting clinical indicators, critical patients |

Each task runs as a full episode: the agent triages all cases sequentially, and the final score is the average reward across all cases.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode. Body: `{"task_id": "easy", "seed": 42}`. Returns first `Observation`. |
| `POST` | `/step` | Submit a triage `Action`. Returns `StepResult` with next observation, reward, and done flag. |
| `GET` | `/state` | Get current environment state (active case, episode progress, last reward). |
| `GET` | `/tasks` | List available tasks with descriptions, case counts, and action schema. |
| `GET` | `/grader` | Get the score from the most recently completed episode. |
| `POST` | `/baseline` | Run LLM baseline inference across all tasks. Requires `OPENAI_API_KEY`. |

## Setup

### Local

```bash
pip install -r requirements.txt
uvicorn app.api:app --port 7860
```

### Docker

```bash
docker build -t medical-triage .
docker run -p 7860:7860 medical-triage
```

## Baseline

Run the LLM baseline to verify end-to-end functionality:

```bash
export OPENAI_API_KEY="your-key-here"
# Start the server, then in another terminal:
curl -X POST http://localhost:7860/baseline
```

The baseline uses `gpt-4o-mini` with temperature 0 and a fixed seed for reproducible results.

### Baseline Scores

| Task | Score |
|---|---|
| Easy | 0.550 |
| Medium | 0.568 |
| Hard | 0.451 |

## Example Usage

```bash
# 1. Reset the environment with the "easy" task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "easy", "seed": 42}'

# 2. Submit a triage action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "urgency_level": 2,
    "department": "cardiology",
    "initial_actions": ["order_ecg", "start_iv", "monitor_vitals"]
  }'

# 3. Check the episode score after completing all cases
curl http://localhost:7860/grader
```

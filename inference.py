"""Baseline inference script for the OpenEnv Medical Triage environment."""

import json
import logging
import os
import sys

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_DEPARTMENTS = [
    "emergency", "cardiology", "orthopedics", "neurology",
    "general_medicine", "pediatrics", "pulmonology",
    "gastroenterology", "dermatology", "psychiatry",
]

VALID_INITIAL_ACTIONS = [
    "start_iv", "order_ecg", "administer_oxygen", "pain_management",
    "order_blood_work", "order_ct_scan", "order_xray",
    "immobilize_limb", "monitor_vitals", "administer_epinephrine",
    "order_mri", "fluid_resuscitation", "apply_tourniquet",
    "administer_nitroglycerin", "intubation_prep",
]


def build_prompt(observation: dict) -> str:
    vitals = observation["vitals"]
    symptoms_str = ", ".join(observation["symptoms"])
    history_str = ", ".join(observation["medical_history"]) if observation["medical_history"] else "None"
    meds_str = ", ".join(observation["current_medications"]) if observation["current_medications"] else "None"

    return (
        f"Triage the following patient and provide your decision.\n\n"
        f"Patient Information:\n"
        f"- Age: {observation['age']}\n"
        f"- Gender: {observation['gender']}\n"
        f"- Chief Complaint: {observation['chief_complaint']}\n"
        f"- Symptoms: {symptoms_str}\n"
        f"- Time of Arrival: {observation['time_of_arrival']}\n\n"
        f"Vitals:\n"
        f"- Heart Rate: {vitals['heart_rate']} bpm\n"
        f"- Blood Pressure: {vitals['blood_pressure']}\n"
        f"- Temperature: {vitals['temperature']} °C\n"
        f"- Oxygen Saturation: {vitals['oxygen_saturation']}%\n"
        f"- Respiratory Rate: {vitals['respiratory_rate']} breaths/min\n\n"
        f"Medical History: {history_str}\n"
        f"Current Medications: {meds_str}\n\n"
        f"Valid departments: {', '.join(VALID_DEPARTMENTS)}\n"
        f"Valid initial actions: {', '.join(VALID_INITIAL_ACTIONS)}\n\n"
        f"Respond with ONLY a JSON object in this exact format:\n"
        f'{{"urgency_level": <int 1-5>, "department": "<str>", '
        f'"initial_actions": ["<str>", ...]}}'
    )


def parse_response(response_text: str) -> dict:
    try:
        text = response_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines[1:] if not l.strip().startswith("```")]
            text = "\n".join(lines)
        data = json.loads(text)
        return {
            "urgency_level": data["urgency_level"],
            "department": data["department"],
            "initial_actions": data.get("initial_actions", []),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Failed to parse LLM response, using fallback")
        return {
            "urgency_level": 3,
            "department": "general_medicine",
            "initial_actions": ["monitor_vitals"],
        }


def run_inference(base_url: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    scores = {}
    task_ids = ["easy", "medium", "hard"]

    with httpx.Client(timeout=120.0) as http:
        for task_id in task_ids:
            logger.info(f"Running task: {task_id}")
            try:
                reset_resp = http.post(f"{base_url}/reset", json={"task_id": task_id, "seed": 42})
                reset_resp.raise_for_status()
                observation = reset_resp.json()

                done = False
                step_count = 0
                while not done:
                    step_count += 1
                    prompt = build_prompt(observation)

                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a medical triage assistant. Respond ONLY with a JSON object, no other text."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0,
                        )
                        text = response.choices[0].message.content or ""
                        action = parse_response(text)
                    except Exception:
                        logger.exception("OpenAI API call failed, using fallback")
                        action = {"urgency_level": 3, "department": "general_medicine", "initial_actions": ["monitor_vitals"]}

                    step_resp = http.post(f"{base_url}/step", json=action)
                    step_resp.raise_for_status()
                    step_data = step_resp.json()

                    done = step_data["done"]
                    if not done:
                        observation = step_data["observation"]

                    logger.info(f"  Step {step_count}: reward={step_data['reward']['total']:.3f}, done={done}")

                grader_resp = http.get(f"{base_url}/grader")
                grader_resp.raise_for_status()
                score = grader_resp.json()["score"]
                scores[task_id] = score
                logger.info(f"Task {task_id} score: {score:.3f}")

            except Exception:
                logger.exception(f"Failed to run task {task_id}")
                scores[task_id] = 0.0

    return scores


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860"
    scores = run_inference(base_url)
    print(json.dumps(scores, indent=2))

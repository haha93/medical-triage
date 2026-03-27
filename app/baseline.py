"""Baseline inference script for the OpenEnv Medical Triage environment."""

import json
import logging
import os

import httpx
from openai import OpenAI

from app.models import Action, Observation, VALID_DEPARTMENTS, VALID_INITIAL_ACTIONS

logger = logging.getLogger(__name__)


class BaselineRunner:
    """Runs baseline LLM inference against the triage environment's HTTP API."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Error: OPENAI_API_KEY environment variable is not set"
            )
        self.client = OpenAI(api_key=api_key)

    def run_all_tasks(self) -> dict[str, float]:
        """Run inference on easy, medium, and hard tasks. Returns scores."""
        scores: dict[str, float] = {}
        task_ids = ["easy", "medium", "hard"]

        with httpx.Client(timeout=120.0) as http:
            for task_id in task_ids:
                try:
                    score = self._run_task(http, task_id)
                    scores[task_id] = score
                except Exception:
                    logger.exception("Failed to run task %s", task_id)
                    scores[task_id] = 0.0

        return scores

    def _run_task(self, http: httpx.Client, task_id: str) -> float:
        """Run a single task through the environment and return its score."""
        # Reset the environment for this task
        reset_resp = http.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": 42},
        )
        reset_resp.raise_for_status()
        observation = Observation(**reset_resp.json())

        done = False
        while not done:
            # Build prompt and get LLM response
            prompt = self._build_prompt(observation)
            action = self._call_llm(prompt)

            # Step the environment
            step_resp = http.post(
                f"{self.base_url}/step",
                json=action.model_dump(),
            )
            step_resp.raise_for_status()
            step_data = step_resp.json()

            done = step_data["done"]
            if not done:
                observation = Observation(**step_data["observation"])

        # Get the episode score
        grader_resp = http.get(f"{self.base_url}/grader")
        grader_resp.raise_for_status()
        return grader_resp.json()["score"]

    def _call_llm(self, prompt: str) -> Action:
        """Send prompt to OpenAI and parse the response into an Action."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a medical triage assistant. Respond ONLY "
                            "with a JSON object, no other text."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            text = response.choices[0].message.content or ""
            return self._parse_response(text)
        except Exception:
            logger.exception("OpenAI API call failed, using fallback action")
            return Action(
                urgency_level=3,
                department="general_medicine",
                initial_actions=["monitor_vitals"],
            )

    def _build_prompt(self, observation: Observation) -> str:
        """Build a structured triage prompt from an Observation."""
        vitals = observation.vitals
        symptoms_str = ", ".join(observation.symptoms)
        history_str = (
            ", ".join(observation.medical_history)
            if observation.medical_history
            else "None"
        )
        meds_str = (
            ", ".join(observation.current_medications)
            if observation.current_medications
            else "None"
        )
        departments_str = ", ".join(VALID_DEPARTMENTS)
        actions_str = ", ".join(VALID_INITIAL_ACTIONS)

        return (
            f"Triage the following patient and provide your decision.\n\n"
            f"Patient Information:\n"
            f"- Age: {observation.age}\n"
            f"- Gender: {observation.gender}\n"
            f"- Chief Complaint: {observation.chief_complaint}\n"
            f"- Symptoms: {symptoms_str}\n"
            f"- Time of Arrival: {observation.time_of_arrival}\n\n"
            f"Vitals:\n"
            f"- Heart Rate: {vitals.heart_rate} bpm\n"
            f"- Blood Pressure: {vitals.blood_pressure}\n"
            f"- Temperature: {vitals.temperature} °C\n"
            f"- Oxygen Saturation: {vitals.oxygen_saturation}%\n"
            f"- Respiratory Rate: {vitals.respiratory_rate} breaths/min\n\n"
            f"Medical History: {history_str}\n"
            f"Current Medications: {meds_str}\n\n"
            f"Valid departments: {departments_str}\n"
            f"Valid initial actions: {actions_str}\n\n"
            f"Respond with ONLY a JSON object in this exact format:\n"
            f'{{"urgency_level": <int 1-5>, "department": "<str>", '
            f'"initial_actions": ["<str>", ...]}}'
        )

    def _parse_response(self, response_text: str) -> Action:
        """Parse LLM response text into an Action, with fallback on failure."""
        try:
            # Strip markdown code fences if present
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last lines (fences)
                lines = [l for l in lines[1:] if not l.strip().startswith("```")]
                text = "\n".join(lines)

            data = json.loads(text)
            return Action(
                urgency_level=data["urgency_level"],
                department=data["department"],
                initial_actions=data.get("initial_actions", []),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "Failed to parse LLM response: %s. Using fallback action. Raw: %s",
                exc,
                response_text[:200],
            )
            return Action(
                urgency_level=3,
                department="general_medicine",
                initial_actions=["monitor_vitals"],
            )

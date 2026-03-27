"""Pydantic models for the OpenEnv Medical Triage environment."""

from typing import Optional

from pydantic import BaseModel, field_validator


VALID_DEPARTMENTS: list[str] = [
    "emergency", "cardiology", "orthopedics", "neurology",
    "general_medicine", "pediatrics", "pulmonology",
    "gastroenterology", "dermatology", "psychiatry",
]

VALID_INITIAL_ACTIONS: list[str] = [
    "start_iv", "order_ecg", "administer_oxygen", "pain_management",
    "order_blood_work", "order_ct_scan", "order_xray",
    "immobilize_limb", "monitor_vitals", "administer_epinephrine",
    "order_mri", "fluid_resuscitation", "apply_tourniquet",
    "administer_nitroglycerin", "intubation_prep",
]


class Vitals(BaseModel):
    heart_rate: int
    blood_pressure: str
    temperature: float
    oxygen_saturation: int
    respiratory_rate: int

    @field_validator("heart_rate")
    @classmethod
    def validate_heart_rate(cls, v: int) -> int:
        if not 30 <= v <= 250:
            raise ValueError("heart_rate must be between 30 and 250")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 35.0 <= v <= 42.0:
            raise ValueError("temperature must be between 35.0 and 42.0")
        return v

    @field_validator("oxygen_saturation")
    @classmethod
    def validate_oxygen_saturation(cls, v: int) -> int:
        if not 50 <= v <= 100:
            raise ValueError("oxygen_saturation must be between 50 and 100")
        return v

    @field_validator("respiratory_rate")
    @classmethod
    def validate_respiratory_rate(cls, v: int) -> int:
        if not 5 <= v <= 60:
            raise ValueError("respiratory_rate must be between 5 and 60")
        return v


class Observation(BaseModel):
    patient_id: str
    symptoms: list[str]
    vitals: Vitals
    medical_history: list[str]
    age: int
    gender: str
    chief_complaint: str
    time_of_arrival: str
    current_medications: list[str]


class Action(BaseModel):
    urgency_level: int
    department: str
    initial_actions: list[str]

    @field_validator("urgency_level")
    @classmethod
    def validate_urgency_level(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("urgency_level must be between 1 and 5")
        return v

    @field_validator("department")
    @classmethod
    def validate_department(cls, v: str) -> str:
        if v not in VALID_DEPARTMENTS:
            raise ValueError(
                f"Invalid department '{v}'. Accepted departments: {VALID_DEPARTMENTS}"
            )
        return v


class Reward(BaseModel):
    total: float
    urgency_score: float
    department_score: float
    actions_score: float
    safety_penalty: float


class GroundTruth(BaseModel):
    urgency_level: int
    department: str
    initial_actions: list[str]
    is_critical: bool

    @field_validator("is_critical", mode="before")
    @classmethod
    def compute_is_critical(cls, v: bool, info) -> bool:
        # Allow explicit value, but also auto-compute from urgency_level
        if v is None and "urgency_level" in info.data:
            return info.data["urgency_level"] in (1, 2)
        return v


class PatientCase(BaseModel):
    observation: Observation
    ground_truth: GroundTruth


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class EnvironmentState(BaseModel):
    current_case: Optional[Observation]
    episode_active: bool
    task_id: Optional[str]
    cases_remaining: int
    last_reward: Optional[Reward]


class TaskInfo(BaseModel):
    task_id: str
    description: str
    difficulty: str
    num_cases: int
    action_schema: dict

"""Predefined patient cases for the OpenEnv Medical Triage environment."""

import random

from app.models import (
    GroundTruth,
    Observation,
    PatientCase,
    Vitals,
    VALID_DEPARTMENTS,
    VALID_INITIAL_ACTIONS,
)


def get_easy_cases() -> list[PatientCase]:
    """Return ~10 easy cases: single symptom, clear department, unambiguous urgency."""
    return [
        PatientCase(
            observation=Observation(
                patient_id="easy-001",
                symptoms=["severe_arm_pain"],
                vitals=Vitals(heart_rate=82, blood_pressure="130/85", temperature=36.8, oxygen_saturation=98, respiratory_rate=16),
                medical_history=[],
                age=35,
                gender="male",
                chief_complaint="Fell off bicycle and landed on arm",
                time_of_arrival="14:30",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=3,
                department="orthopedics",
                initial_actions=["pain_management", "order_xray", "immobilize_limb"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-002",
                symptoms=["skin_rash"],
                vitals=Vitals(heart_rate=72, blood_pressure="120/78", temperature=36.6, oxygen_saturation=99, respiratory_rate=14),
                medical_history=[],
                age=28,
                gender="female",
                chief_complaint="Itchy red rash on both arms for two days",
                time_of_arrival="10:15",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=5,
                department="dermatology",
                initial_actions=["monitor_vitals"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-003",
                symptoms=["chest_pain"],
                vitals=Vitals(heart_rate=110, blood_pressure="155/95", temperature=36.9, oxygen_saturation=96, respiratory_rate=20),
                medical_history=[],
                age=62,
                gender="male",
                chief_complaint="Sudden onset chest tightness while resting",
                time_of_arrival="08:45",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="cardiology",
                initial_actions=["order_ecg", "start_iv", "administer_nitroglycerin", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-004",
                symptoms=["high_fever"],
                vitals=Vitals(heart_rate=100, blood_pressure="118/75", temperature=39.5, oxygen_saturation=97, respiratory_rate=18),
                medical_history=[],
                age=5,
                gender="female",
                chief_complaint="Child has had a high fever since last night",
                time_of_arrival="07:00",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=3,
                department="pediatrics",
                initial_actions=["monitor_vitals", "order_blood_work"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-005",
                symptoms=["severe_headache"],
                vitals=Vitals(heart_rate=78, blood_pressure="140/88", temperature=36.7, oxygen_saturation=99, respiratory_rate=15),
                medical_history=[],
                age=45,
                gender="female",
                chief_complaint="Worst headache of my life, came on suddenly",
                time_of_arrival="22:10",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="neurology",
                initial_actions=["order_ct_scan", "start_iv", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-006",
                symptoms=["abdominal_pain"],
                vitals=Vitals(heart_rate=88, blood_pressure="125/80", temperature=37.2, oxygen_saturation=98, respiratory_rate=16),
                medical_history=[],
                age=30,
                gender="male",
                chief_complaint="Sharp pain in lower right abdomen for six hours",
                time_of_arrival="16:20",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=3,
                department="gastroenterology",
                initial_actions=["pain_management", "order_blood_work", "order_ct_scan"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-007",
                symptoms=["shortness_of_breath"],
                vitals=Vitals(heart_rate=105, blood_pressure="130/82", temperature=36.5, oxygen_saturation=88, respiratory_rate=28),
                medical_history=[],
                age=70,
                gender="male",
                chief_complaint="Difficulty breathing that started this morning",
                time_of_arrival="11:30",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="pulmonology",
                initial_actions=["administer_oxygen", "order_xray", "monitor_vitals", "start_iv"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-008",
                symptoms=["anxiety_attack"],
                vitals=Vitals(heart_rate=115, blood_pressure="145/90", temperature=36.8, oxygen_saturation=99, respiratory_rate=24),
                medical_history=[],
                age=25,
                gender="female",
                chief_complaint="Feeling of impending doom, heart racing, cannot calm down",
                time_of_arrival="19:45",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=4,
                department="psychiatry",
                initial_actions=["monitor_vitals"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-009",
                symptoms=["laceration"],
                vitals=Vitals(heart_rate=90, blood_pressure="128/82", temperature=36.7, oxygen_saturation=98, respiratory_rate=16),
                medical_history=[],
                age=40,
                gender="male",
                chief_complaint="Deep cut on hand from kitchen knife",
                time_of_arrival="18:00",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=4,
                department="emergency",
                initial_actions=["pain_management", "monitor_vitals"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="easy-010",
                symptoms=["mild_cough"],
                vitals=Vitals(heart_rate=75, blood_pressure="118/76", temperature=37.0, oxygen_saturation=98, respiratory_rate=15),
                medical_history=[],
                age=50,
                gender="female",
                chief_complaint="Persistent dry cough for one week",
                time_of_arrival="09:30",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=5,
                department="general_medicine",
                initial_actions=["monitor_vitals"],
                is_critical=False,
            ),
        ),
    ]


def get_medium_cases() -> list[PatientCase]:
    """Return ~10 medium cases: multiple symptoms, some ambiguity, requires reasoning."""
    return [
        PatientCase(
            observation=Observation(
                patient_id="medium-001",
                symptoms=["chest_pain", "shortness_of_breath"],
                vitals=Vitals(heart_rate=118, blood_pressure="160/100", temperature=36.9, oxygen_saturation=93, respiratory_rate=22),
                medical_history=[],
                age=58,
                gender="male",
                chief_complaint="Chest tightness with difficulty breathing after climbing stairs",
                time_of_arrival="09:15",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="cardiology",
                initial_actions=["order_ecg", "start_iv", "administer_oxygen", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-002",
                symptoms=["severe_headache", "blurred_vision"],
                vitals=Vitals(heart_rate=85, blood_pressure="180/110", temperature=36.8, oxygen_saturation=98, respiratory_rate=16),
                medical_history=[],
                age=55,
                gender="female",
                chief_complaint="Sudden severe headache with vision changes",
                time_of_arrival="15:40",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="neurology",
                initial_actions=["order_ct_scan", "start_iv", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-003",
                symptoms=["abdominal_pain", "nausea", "fever"],
                vitals=Vitals(heart_rate=95, blood_pressure="130/85", temperature=38.5, oxygen_saturation=97, respiratory_rate=18),
                medical_history=[],
                age=32,
                gender="female",
                chief_complaint="Severe stomach pain with nausea and fever since yesterday",
                time_of_arrival="06:30",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=3,
                department="gastroenterology",
                initial_actions=["pain_management", "order_blood_work", "order_ct_scan", "start_iv"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-004",
                symptoms=["wheezing", "shortness_of_breath"],
                vitals=Vitals(heart_rate=108, blood_pressure="135/85", temperature=37.1, oxygen_saturation=90, respiratory_rate=26),
                medical_history=[],
                age=12,
                gender="male",
                chief_complaint="Asthma-like episode, trouble breathing and wheezing",
                time_of_arrival="21:00",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="pediatrics",
                initial_actions=["administer_oxygen", "monitor_vitals", "start_iv"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-005",
                symptoms=["leg_pain", "swelling"],
                vitals=Vitals(heart_rate=92, blood_pressure="138/88", temperature=37.3, oxygen_saturation=97, respiratory_rate=17),
                medical_history=[],
                age=65,
                gender="female",
                chief_complaint="Left leg is swollen, red, and painful to touch",
                time_of_arrival="13:20",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=3,
                department="emergency",
                initial_actions=["order_blood_work", "monitor_vitals", "pain_management"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-006",
                symptoms=["dizziness", "fainting"],
                vitals=Vitals(heart_rate=55, blood_pressure="95/60", temperature=36.5, oxygen_saturation=96, respiratory_rate=14),
                medical_history=[],
                age=72,
                gender="male",
                chief_complaint="Fainted twice today and feeling very dizzy",
                time_of_arrival="11:00",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="cardiology",
                initial_actions=["order_ecg", "start_iv", "monitor_vitals", "order_blood_work"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-007",
                symptoms=["back_pain", "numbness_in_legs"],
                vitals=Vitals(heart_rate=80, blood_pressure="125/80", temperature=36.7, oxygen_saturation=98, respiratory_rate=15),
                medical_history=[],
                age=48,
                gender="male",
                chief_complaint="Severe lower back pain with tingling and numbness in both legs",
                time_of_arrival="17:45",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=3,
                department="neurology",
                initial_actions=["pain_management", "order_mri", "monitor_vitals"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-008",
                symptoms=["coughing_blood", "chest_tightness"],
                vitals=Vitals(heart_rate=100, blood_pressure="140/90", temperature=37.5, oxygen_saturation=92, respiratory_rate=22),
                medical_history=[],
                age=60,
                gender="male",
                chief_complaint="Coughing up small amounts of blood with chest tightness",
                time_of_arrival="08:00",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="pulmonology",
                initial_actions=["order_ct_scan", "order_blood_work", "administer_oxygen", "start_iv", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-009",
                symptoms=["high_fever", "sore_throat", "body_aches"],
                vitals=Vitals(heart_rate=102, blood_pressure="122/78", temperature=39.8, oxygen_saturation=96, respiratory_rate=19),
                medical_history=[],
                age=8,
                gender="female",
                chief_complaint="Child with high fever, sore throat, and body aches for two days",
                time_of_arrival="20:30",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=3,
                department="pediatrics",
                initial_actions=["order_blood_work", "monitor_vitals"],
                is_critical=False,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="medium-010",
                symptoms=["confusion", "slurred_speech"],
                vitals=Vitals(heart_rate=88, blood_pressure="170/105", temperature=36.9, oxygen_saturation=97, respiratory_rate=16),
                medical_history=[],
                age=68,
                gender="male",
                chief_complaint="Wife noticed sudden confusion and difficulty speaking",
                time_of_arrival="05:15",
                current_medications=[],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="neurology",
                initial_actions=["order_ct_scan", "start_iv", "monitor_vitals", "order_blood_work"],
                is_critical=True,
            ),
        ),
    ]


def get_hard_cases() -> list[PatientCase]:
    """Return ~10 hard cases: complex multi-condition, conflicting indicators, critical patients."""
    return [
        PatientCase(
            observation=Observation(
                patient_id="hard-001",
                symptoms=["chest_pain", "shortness_of_breath", "diaphoresis"],
                vitals=Vitals(heart_rate=130, blood_pressure="85/55", temperature=36.4, oxygen_saturation=87, respiratory_rate=30),
                medical_history=["previous_myocardial_infarction", "type_2_diabetes", "hypertension"],
                age=71,
                gender="male",
                chief_complaint="Crushing chest pain radiating to jaw, profuse sweating, feels like previous heart attack",
                time_of_arrival="03:20",
                current_medications=["metformin", "lisinopril", "aspirin"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="cardiology",
                initial_actions=["order_ecg", "start_iv", "administer_nitroglycerin", "administer_oxygen", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-002",
                symptoms=["severe_headache", "neck_stiffness", "photophobia", "high_fever"],
                vitals=Vitals(heart_rate=112, blood_pressure="145/92", temperature=40.1, oxygen_saturation=96, respiratory_rate=22),
                medical_history=["immunosuppression", "kidney_transplant"],
                age=45,
                gender="female",
                chief_complaint="Worst headache ever with stiff neck, cannot tolerate light, high fever",
                time_of_arrival="01:45",
                current_medications=["tacrolimus", "prednisone"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="emergency",
                initial_actions=["start_iv", "order_ct_scan", "order_blood_work", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-003",
                symptoms=["abdominal_pain", "vomiting_blood", "dizziness"],
                vitals=Vitals(heart_rate=120, blood_pressure="90/58", temperature=36.6, oxygen_saturation=94, respiratory_rate=24),
                medical_history=["liver_cirrhosis", "chronic_alcohol_use", "esophageal_varices"],
                age=56,
                gender="male",
                chief_complaint="Vomiting large amounts of blood, severe abdominal pain, feeling faint",
                time_of_arrival="23:30",
                current_medications=["propranolol", "lactulose"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="gastroenterology",
                initial_actions=["start_iv", "fluid_resuscitation", "order_blood_work", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-004",
                symptoms=["shortness_of_breath", "leg_swelling", "fatigue"],
                vitals=Vitals(heart_rate=105, blood_pressure="100/65", temperature=36.5, oxygen_saturation=85, respiratory_rate=28),
                medical_history=["congestive_heart_failure", "atrial_fibrillation", "copd"],
                age=78,
                gender="female",
                chief_complaint="Worsening shortness of breath over three days, legs very swollen, cannot lie flat",
                time_of_arrival="04:00",
                current_medications=["furosemide", "warfarin", "digoxin", "albuterol"],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="cardiology",
                initial_actions=["administer_oxygen", "start_iv", "order_ecg", "order_xray", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-005",
                symptoms=["confusion", "tremors", "slurred_speech"],
                vitals=Vitals(heart_rate=98, blood_pressure="150/88", temperature=36.3, oxygen_saturation=97, respiratory_rate=18),
                medical_history=["type_1_diabetes", "epilepsy", "hypothyroidism"],
                age=34,
                gender="male",
                chief_complaint="Found confused and shaking, speech is garbled, missed meals today",
                time_of_arrival="12:15",
                current_medications=["insulin", "levetiracetam", "levothyroxine"],
            ),
            ground_truth=GroundTruth(
                urgency_level=2,
                department="emergency",
                initial_actions=["start_iv", "order_blood_work", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-006",
                symptoms=["wheezing", "cough", "chest_tightness", "cyanosis"],
                vitals=Vitals(heart_rate=125, blood_pressure="140/90", temperature=37.0, oxygen_saturation=82, respiratory_rate=35),
                medical_history=["severe_asthma", "previous_intubation", "gerd"],
                age=22,
                gender="female",
                chief_complaint="Severe asthma attack not responding to home inhaler, lips turning blue",
                time_of_arrival="02:30",
                current_medications=["fluticasone", "albuterol", "omeprazole"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="pulmonology",
                initial_actions=["administer_oxygen", "start_iv", "monitor_vitals", "intubation_prep"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-007",
                symptoms=["severe_abdominal_pain", "fever", "rigid_abdomen"],
                vitals=Vitals(heart_rate=115, blood_pressure="95/60", temperature=39.2, oxygen_saturation=95, respiratory_rate=24),
                medical_history=["peptic_ulcer_disease", "nsaid_overuse", "smoking"],
                age=52,
                gender="male",
                chief_complaint="Sudden severe abdominal pain, abdomen is board-like rigid, fever climbing",
                time_of_arrival="07:45",
                current_medications=["omeprazole", "ibuprofen"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="emergency",
                initial_actions=["start_iv", "fluid_resuscitation", "order_ct_scan", "order_blood_work", "monitor_vitals", "pain_management"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-008",
                symptoms=["weakness", "numbness", "facial_droop", "difficulty_speaking"],
                vitals=Vitals(heart_rate=88, blood_pressure="185/110", temperature=36.8, oxygen_saturation=96, respiratory_rate=17),
                medical_history=["atrial_fibrillation", "hypertension", "previous_tia"],
                age=66,
                gender="female",
                chief_complaint="Sudden right-sided weakness, face drooping on one side, cannot form words",
                time_of_arrival="06:10",
                current_medications=["apixaban", "amlodipine", "atorvastatin"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="neurology",
                initial_actions=["order_ct_scan", "start_iv", "monitor_vitals", "order_blood_work"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-009",
                symptoms=["swollen_face", "difficulty_breathing", "hives"],
                vitals=Vitals(heart_rate=130, blood_pressure="80/50", temperature=36.9, oxygen_saturation=89, respiratory_rate=30),
                medical_history=["peanut_allergy", "previous_anaphylaxis", "asthma"],
                age=16,
                gender="male",
                chief_complaint="Ate something with peanuts, face swelling rapidly, trouble breathing, covered in hives",
                time_of_arrival="12:45",
                current_medications=["montelukast"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="emergency",
                initial_actions=["administer_epinephrine", "administer_oxygen", "start_iv", "fluid_resuscitation", "monitor_vitals"],
                is_critical=True,
            ),
        ),
        PatientCase(
            observation=Observation(
                patient_id="hard-010",
                symptoms=["chest_pain", "back_pain", "tearing_sensation"],
                vitals=Vitals(heart_rate=110, blood_pressure="200/115", temperature=36.7, oxygen_saturation=94, respiratory_rate=22),
                medical_history=["marfan_syndrome", "hypertension", "aortic_root_dilation"],
                age=42,
                gender="male",
                chief_complaint="Sudden severe tearing pain in chest radiating to back, feels like something ripping",
                time_of_arrival="10:00",
                current_medications=["losartan", "metoprolol"],
            ),
            ground_truth=GroundTruth(
                urgency_level=1,
                department="cardiology",
                initial_actions=["start_iv", "order_ct_scan", "pain_management", "monitor_vitals", "order_blood_work"],
                is_critical=True,
            ),
        ),
    ]


def get_cases_for_task(task_id: str, seed: int) -> list[PatientCase]:
    """Return deterministic ordered list of cases for a task.

    Args:
        task_id: One of "easy", "medium", "hard".
        seed: Random seed for deterministic shuffling.

    Returns:
        Shuffled list of PatientCase objects for the given task.

    Raises:
        ValueError: If task_id is not one of the valid task IDs.
    """
    case_loaders = {
        "easy": get_easy_cases,
        "medium": get_medium_cases,
        "hard": get_hard_cases,
    }

    if task_id not in case_loaders:
        raise ValueError(
            f"Unknown task_id '{task_id}'. Available tasks: {list(case_loaders.keys())}"
        )

    cases = case_loaders[task_id]()
    rng = random.Random(seed)
    rng.shuffle(cases)
    return cases

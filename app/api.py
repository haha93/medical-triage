"""FastAPI server for the OpenEnv Medical Triage environment."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.models import Action, Observation, StepResult, EnvironmentState, TaskInfo
from app.environment import TriageEnvironment

app = FastAPI(title="OpenEnv Medical Triage", version="1.0.0")
env = TriageEnvironment()


class ResetRequest(BaseModel):
    task_id: str
    seed: int = 42


@app.post("/reset", response_model=Observation)
def reset(request: ResetRequest) -> Observation:
    """Reset the environment with a new task and return the first observation."""
    try:
        return env.reset(request.task_id, request.seed)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail="Unknown task_id. Available tasks: easy, medium, hard",
        )


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    """Submit a triage action and receive the result."""
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    """Return the current environment state snapshot."""
    return env.state()


@app.get("/tasks", response_model=list[TaskInfo])
def tasks() -> list[TaskInfo]:
    """Return available tasks with their metadata and action schema."""
    action_schema = Action.model_json_schema()
    return [
        TaskInfo(
            task_id="easy",
            description="Single clear symptom, obvious triage decisions",
            difficulty="easy",
            num_cases=10,
            action_schema=action_schema,
        ),
        TaskInfo(
            task_id="medium",
            description="Multiple symptoms with diagnostic ambiguity",
            difficulty="medium",
            num_cases=10,
            action_schema=action_schema,
        ),
        TaskInfo(
            task_id="hard",
            description="Complex multi-condition patients with conflicting indicators",
            difficulty="hard",
            num_cases=10,
            action_schema=action_schema,
        ),
    ]


@app.get("/grader")
def grader() -> dict:
    """Return the score from the most recently completed episode."""
    score = env.get_episode_score()
    if score is None:
        raise HTTPException(
            status_code=400,
            detail="No completed episode available. Complete an episode first.",
        )
    return {"score": score}


@app.post("/baseline")
def baseline() -> dict:
    """Run baseline inference across all tasks and return scores."""
    try:
        from app.baseline import BaselineRunner

        runner = BaselineRunner()
        scores = runner.run_all_tasks()
        return scores
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Baseline execution failed: {str(e)}",
        )

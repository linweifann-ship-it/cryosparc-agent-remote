# Defines the model decision schema and validation result models.
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


DecisionType = Literal["forward", "rollback", "branch", "stop", "request_input"]
ActionType = Literal["forward", "branch"]
RollbackMode = Literal[
    "mark_only",
    "rerun_from_target",
    "branch_from_target",
    "manual_review",
]


# Model-selected actions may either reference an MCP candidate or name a job directly.
class Action(BaseModel):
    action_id: str | None = None
    action_type: ActionType = "forward"
    workflow_node_id: str | None = None
    job_type: str
    parameters: dict[str, Any]
    connections: dict[str, Any] | None = None


class RollbackTarget(BaseModel):
    workflow_node_id: str
    job_type: str
    reason_code: str
    rollback_mode: RollbackMode | None = None


class BranchPlan(BaseModel):
    branch_type: str
    max_parallel_branches: int = Field(ge=1)
    notes: str | None = None


# Top-level model output contract for the upstream decision model.
class ModelDecision(BaseModel):
    schema_version: Literal["1.0"]
    state_snapshot_id: str | None = None
    candidate_set_id: str | None = None
    decision_type: DecisionType
    selected_actions: list[Action]
    rollback_target: RollbackTarget | None = None
    branch_plan: BranchPlan | None = None
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    risk_flags: list[str]
    evidence: list[str]
    requested_inputs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "ModelDecision":
        if self.decision_type == "forward":
            if not self.selected_actions:
                raise ValueError("forward decisions require at least one selected_action")
            if self.rollback_target is not None:
                raise ValueError("forward decisions must not include rollback_target")
            if self.branch_plan is not None:
                raise ValueError("forward decisions must not include branch_plan")

        if self.decision_type == "branch":
            if not self.selected_actions:
                raise ValueError("branch decisions require at least one selected_action")
            if self.branch_plan is None:
                raise ValueError("branch decisions require branch_plan")
            if self.rollback_target is not None:
                raise ValueError("branch decisions must not include rollback_target")

        if self.decision_type == "rollback":
            if self.selected_actions:
                raise ValueError("rollback decisions must not include selected_actions")
            if self.rollback_target is None:
                raise ValueError("rollback decisions require rollback_target")
            if self.branch_plan is not None:
                raise ValueError("rollback decisions must not include branch_plan")

        if self.decision_type == "stop":
            if self.selected_actions:
                raise ValueError("stop decisions must not include selected_actions")
            if self.rollback_target is not None:
                raise ValueError("stop decisions must not include rollback_target")
            if self.branch_plan is not None:
                raise ValueError("stop decisions must not include branch_plan")

        if self.decision_type == "request_input":
            if self.selected_actions:
                raise ValueError("request_input decisions must not include selected_actions")
            if self.rollback_target is not None:
                raise ValueError("request_input decisions must not include rollback_target")
            if self.branch_plan is not None:
                raise ValueError("request_input decisions must not include branch_plan")
            if not self.requested_inputs:
                raise ValueError("request_input decisions require requested_inputs")

        for action in self.selected_actions:
            if self.decision_type == "forward" and action.action_type != "forward":
                raise ValueError("forward decisions can only include forward actions")
            if self.decision_type == "branch" and action.action_type != "branch":
                raise ValueError("branch decisions can only include branch actions")

        return self


class ValidationIssue(BaseModel):
    severity: Literal["error", "warning"] = "error"
    code: str
    message: str
    path: str | None = None


# Structured validation outputs are returned directly to MCP callers.
class ResolvedAction(BaseModel):
    action_id: str
    action_type: str
    workflow_node_id: str
    job_type: str
    execution_mode: str
    resolved_parameters: dict[str, Any]
    connections: dict[str, Any] | None = None
    mcp_tool_name: str | None = None


class ValidationResult(BaseModel):
    success: bool
    valid_schema: bool
    valid_actions: bool
    decision_type: str | None = None
    dry_run: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)
    resolved_actions: list[ResolvedAction] = Field(default_factory=list)


def parse_model_decision(payload: dict[str, Any]) -> tuple[ModelDecision | None, list[ValidationIssue]]:
    """Parse a raw JSON payload into ModelDecision or normalized issues."""
    try:
        return ModelDecision.model_validate(payload), []
    except ValidationError as exc:
        issues = []
        for error in exc.errors():
            path = ".".join(str(part) for part in error.get("loc", ())) or None
            issues.append(
                ValidationIssue(
                    code="schema_validation_error",
                    message=error.get("msg", "Invalid model decision schema"),
                    path=path,
                )
            )
        return None, issues
    except ValueError as exc:
        return None, [ValidationIssue(code="schema_validation_error", message=str(exc))]

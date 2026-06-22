from pydantic import BaseModel, Field
from typing import Optional, Literal

Tier = Literal["T1","T2","CHECKER"]
Status = Literal["open","in_progress","pending_approval","approved","rejected","closed"]

class CaseOut(BaseModel):
    case_id: str
    title: Optional[str] = None
    status: Status
    tier: Tier
    risk_score: float = 0.0
    entity_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    assigned_to: Optional[str] = None
    submitted_by: Optional[str] = None
    submitted_at: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    approval_comment: Optional[str] = None

class AssignIn(BaseModel):
    actor: str
    assigned_to: str

class SubmitIn(BaseModel):
    actor: str

class ApproveRejectIn(BaseModel):
    actor: str
    comment: str = Field(min_length=1)
    decision: Literal["approved","rejected"]

class NoteIn(BaseModel):
    actor: str
    note: str = Field(min_length=1)

class ListQuery(BaseModel):
    tier: Optional[Tier] = None
    status: Optional[Status] = None
    assigned_to: Optional[str] = None
    unassigned_only: bool = False
    limit: int = 200
    offset: int = 0

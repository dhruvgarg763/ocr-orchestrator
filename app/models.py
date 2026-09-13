from pydantic import BaseModel
from typing import Any


class JobResponse(BaseModel):
    job_id: str
    status: str
    total_pages: int


class PageState(BaseModel):
    job_id: str
    page_number: int
    status: str
    attempt: int = 0
    model: str | None = None
    result: dict[str, Any] | None = None
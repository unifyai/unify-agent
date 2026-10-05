from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, create_model, field_validator, model_validator

from unify.common.stale_reason import StaleReason, coerce_stale_reasons

UNASSIGNED = -1


class Guidance(BaseModel):
    guidance_id: int = Field(
        default=UNASSIGNED,
        description="Auto-incrementing unique identifier for the guidance entry",
        ge=UNASSIGNED,
    )
    title: str = Field(
        description="Short-form title of the guidance (a few words)",
        min_length=1,
        max_length=200,
        json_schema_extra={"ui_editable": True},
    )
    content: str = Field(
        description="Full description of the guidance",
        min_length=1,
        json_schema_extra={"ui_editable": True},
    )
    function_ids: List[int] = Field(
        default_factory=list,
        description=(
            "List of Function.function_id values that this guidance is relevant for. "
            "Represents a many-to-many relationship between Guidance and Functions."
        ),
        json_schema_extra={"ui_editable": True},
    )
    stale_reasons: List[StaleReason] = Field(
        default_factory=list,
        description="Structured records for related functions that no longer resolve.",
    )

    is_builtin: bool = Field(
        default=False,
        description=(
            "True for a read-only builtin entry from the seeded catalogue; "
            "False for an entry the user's own assistant stored."
        ),
    )

    @field_validator("is_builtin", mode="before")
    @classmethod
    def _validate_is_builtin(cls, v):
        if v is None:
            return False
        return v

    @field_validator("function_ids", mode="before")
    @classmethod
    def _validate_function_ids(cls, v):
        """Ensure function_ids is a list[int]. None → []. Coerce values to int."""
        if v is None:
            return []
        if not isinstance(v, list):
            raise TypeError("function_ids must be a list[int]")
        out: list[int] = []
        for item in v:
            try:
                out.append(int(item))
            except Exception as exc:
                raise ValueError("function_ids must contain integers") from exc
        return out

    @field_validator("stale_reasons", mode="before")
    @classmethod
    def _validate_stale_reasons(cls, v):
        return coerce_stale_reasons(v)

    @model_validator(mode="before")
    @classmethod
    def _inject_sentinel(cls, data: dict) -> dict:
        data.setdefault("guidance_id", UNASSIGNED)
        return data

    def to_post_json(self) -> dict:
        exclude = {"guidance_id"} if self.guidance_id == UNASSIGNED else set()
        return self.model_dump(mode="json", exclude=exclude)


# UNIFY_GUIDANCE_LINKED_NAMES: a read of an entry, also naming the functions
# it links. Still named ``Guidance`` (its repr, and the type a sandboxed
# worker reports); built only by the guidance manager's reads.
GuidanceWithLinks = create_model(
    "Guidance",
    __base__=Guidance,
    linked_functions=(
        List[str],
        Field(
            default_factory=list,
            description=(
                "The stored functions function_ids names, each as "
                "`name(signature)`, `(async)` after an async def."
            ),
        ),
    ),
)


# UNIFY_ENTRY_RECORD: a read of an entry with its record (what it was written
# for, its status, its use and the functions linked to it). Still named
# ``Guidance``.
_RECORD_FIELDS = dict(
    record=(
        str,
        Field(
            default="",
            description=(
                "What the entry was written for, how that session ended, its "
                "status and how it has been used."
            ),
        ),
    ),
)
GuidanceWithRecord = create_model("Guidance", __base__=Guidance, **_RECORD_FIELDS)
GuidanceWithLinksAndRecord = create_model(
    "Guidance",
    __base__=GuidanceWithLinks,
    **_RECORD_FIELDS,
)

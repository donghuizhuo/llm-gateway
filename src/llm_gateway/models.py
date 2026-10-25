from pydantic import BaseModel, ConfigDict, Field, field_validator


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    question: str = Field(
        min_length=1, max_length=4000, description="The question to ask the model."
    )
    max_output_tokens: int = Field(
        default=512,
        ge=1,
        le=2048,
        description="The maximum number of tokens to generate in the output.",
    )
    model: str = Field(
        default="gpt-4.1-mini",
        description="The model to use for generating the answer.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty or whitespace.")
        return value

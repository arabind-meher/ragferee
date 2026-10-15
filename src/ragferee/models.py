"""Domain models."""

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A reference to a location in the Laws of the Game."""

    model_config = ConfigDict(frozen=True)

    law_number: int = Field(ge=1, le=17)
    section: str = Field(min_length=1)
    page: int = Field(ge=1)

    def __str__(self) -> str:
        return f"Law {self.law_number}, {self.section}, p. {self.page}"


class Chunk(BaseModel):
    """A passage of Law text with its location in the Laws of the Game."""

    model_config = ConfigDict(frozen=True)

    law_number: int = Field(ge=1, le=17)
    law_title: str = Field(min_length=1)
    section: str = Field(min_length=1)
    page: int = Field(ge=1)
    text: str = Field(min_length=1)

    def citation(self) -> Citation:
        return Citation(law_number=self.law_number, section=self.section, page=self.page)


class Answer(BaseModel):
    """A plain-language answer with the citations that support it."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    citations: tuple[Citation, ...]

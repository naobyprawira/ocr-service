"""
Pydantic schemas for request and response payloads.

These models ensure that data returned by the API conforms to a
predictable structure. They also provide automatic documentation via
FastAPI's generated OpenAPI specification.
"""
from typing import List
from pydantic import BaseModel


class PageContent(BaseModel):
    index: int
    text: str


class OcrLikeResponse(BaseModel):
    doc: dict
    pages: List[PageContent]
    text: str
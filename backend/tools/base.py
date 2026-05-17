from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel


class ToolSchema(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = {}
    risk_level: Literal["low", "medium", "high"] = "low"


class BaseTool(ABC):
    """
    MCP-shaped base class for all Friday tools.
    Each tool has a name, schema, and an async execute method.
    When migrating to MCP, each subclass maps 1-to-1 to an MCP Tool definition.
    """

    schema: ToolSchema

    @abstractmethod
    async def execute(self, params: dict) -> Any:
        """Execute the tool with the given params. Must be overridden."""
        ...

    def describe(self) -> dict:
        """Return a serialisable description of this tool (MCP-compatible)."""
        return self.schema.model_dump()

    def __repr__(self) -> str:
        return f"<Tool name={self.schema.name!r} risk={self.schema.risk_level!r}>"

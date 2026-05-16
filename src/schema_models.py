"""
Pydantic templates for validating and structuring function schemas.
Ensures that input definitions are strict and safe.
"""

from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class ParameterDef(BaseModel):
    """Defining an individual parameter of a function"""
    type: str = Field(..., description="The parameter type (e.g., 'number', 'string')")


class FunctionDef(BaseModel):
    """Complete definition of a function available for the LLM"""
    name: str = Field(..., description="Name of function (e.g., 'fn_add_numbers')")
    description: str = Field(..., description="Description of what the function does.")
    parameters: Dict[str, ParameterDef] = Field(
        default_factory=dict,
        description="Dictionary of expected parameters"
    )

    returns: Optional[Dict[str, Any]] = None

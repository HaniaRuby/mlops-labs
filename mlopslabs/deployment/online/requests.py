from typing import List
from pydantic import BaseModel, Field


# 1. Individual passenger data structure contract
class PassengerRecord(BaseModel):
    Pclass: int = Field(..., description="Ticket class (1, 2, 3)")
    Age: float = Field(..., description="Age of passenger")
    SibSp: int = Field(..., description="Number of siblings/spouses aboard")
    Parch: int = Field(..., description="Number of parents/children aboard")
    Fare: float = Field(..., description="Passenger fare")
    Sex: str = Field(..., description="male or female")
    Embarked: str = Field(..., description="Port of Embarkation (C, Q, S)")


# 2. Batch wrapper matching her exact wrapper structure: request["input"]
# This allows a single request payload to contain a list of multiple records!
class InferenceRequest(BaseModel):
    input: List[PassengerRecord]
from abc import abstractmethod
from typing import Optional
from pydantic import BaseModel
from enum import Enum

class BiasType(str, Enum):
    NEUTRAL = "neutral"
    SHORT = "short"
    LONG = "long"

class BiasRequest(BaseModel):
    symbol: str

class BiasResponse(BaseModel):
    bias: BiasType
    error: Optional[str] = None
    usedSymbol: bool = False

class BiasInterface():
    ignore = False
    @abstractmethod
    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        raise NotImplementedError("Method not implemented")


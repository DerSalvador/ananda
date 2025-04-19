from functools import cache
import os
import importlib
import inspect
import shutil
from constants import DEFAULT_CONFIG
from utils import get_logger
from tinydb import TinyDB, Query
from abc import abstractmethod
from typing import Optional
from pydantic import BaseModel
from enum import Enum


logger = get_logger()
CONFIG_PATH = os.getenv("CONFIG_PATH", "/tmp")

biasdb = TinyDB(f"{CONFIG_PATH}/biasdb.json")
# Entries of type {'name': 'CoinGeckoBTC', 'active': True} def init():

def get_biases():
    biases = biasdb.all()
    all_names = getAllInterfaceNames()
    filtered_biases = []
    for bias in biases:
        if bias["name"] in all_names:
            filtered_biases.append(bias)
    logger.info(f"Getting biases: {biases}")
    return filtered_biases

def get_bias(bias):
    return biasdb.search(Query().name == bias)

def update_bias(bias, active=True):
    # if doesn't exist, create
    if not biasdb.search(Query().name == bias):
        biasdb.insert({"name": bias, "active": active, "paid": False})
    else:
        biasdb.update({"active": active}, Query().name == bias)
    getInterfaces.cache_clear()

def update_config(configname, configvalue):
    table = biasdb.table("configs")
    get_config.cache_clear()
    table.upsert({"name": configname, "value": configvalue}, Query().name == configname)

@cache
def get_config(configname, default=None):
    table = biasdb.table("configs")
    config = table.get(Query().name == configname)
    if config:
        return config["value"]
    return default

def get_all_configs():
    table = biasdb.table("configs")
    all_configs = table.all()
    filtered_configs = []
    for config in all_configs:
        if config["name"] in DEFAULT_CONFIG:
            filtered_configs.append(config)
    return filtered_configs

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

@cache
def getAllInterfaceNames():
    # List all Python files in the directory
    interfaces = []
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "interface.py"]:
            try:
                module_name = f"bias.{filename[:-3]}"
                module = importlib.import_module(module_name)

                # Inspect the module for classes implementing BiasInterface
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BiasInterface) and obj is not BiasInterface:
                        interfaces.append(name)
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    logger.info(f"Getting interfaces: {interfaces}")
    return interfaces

@cache
def getInterfaces(all = False):
    # List all Python files in the directory
    interfaces = {}
    for filename in os.listdir(current_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "interface.py"]:
            try:
                module_name = f"bias.{filename[:-3]}"
                module = importlib.import_module(module_name)

                # Inspect the module for classes implementing BiasInterface
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BiasInterface) and obj is not BiasInterface:
                        # interface_name = filename[:-3]
                        # interface_name = obj.__name__
                        interface_name = name
                        existing_bias = get_bias(interface_name)
                        if existing_bias:
                            if not existing_bias[0]["active"]:
                                obj.ignore = True
                            else:
                                obj.ignore = False
                        if not obj.ignore or all:
                            interfaces[interface_name] = obj()
                        else:
                            logger.info(f"Ignoring {interface_name}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    logger.info(f"Getting interfaces: {list(interfaces.keys())}")
    return interfaces

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
    reason: str = ""
    weight: float = 1.0

class BiasInterface():
    ignore = False
    paid = False
    @abstractmethod
    def bias(self, biasRequest: BiasRequest) -> BiasResponse:
        raise NotImplementedError("Method not implemented")

    def bias_wrapper(self, biasRequest: BiasRequest) -> BiasResponse:
        try:
            paid_weight = float(get_config("BiasWeightPaid", 1.0))
            free_weight = float(get_config("BiasWeightFree", 1.0))
            biasResponse = self.bias(biasRequest)
            biasResponse.weight = paid_weight if self.paid else free_weight
            return biasResponse
        except Exception as e:
            return BiasResponse(bias=BiasType.NEUTRAL, error=str(e))


def init():
    try:
        biasdb.all()
    except Exception as e:
        logger.error(f"Error loading biasdb: {e}, DELETING and RECREATING")
        os.remove(f"{CONFIG_PATH}/biasdb.json")
    interfaces = getInterfaces(all=True)
    for bias in interfaces.keys():
        if not biasdb.search(Query().name == bias):
            biasdb.insert({"name": bias, "active": True})
        else:
            biasdb.update({"paid": interfaces[bias].paid}, Query().name == bias)
    configs = DEFAULT_CONFIG
    for name, value in configs.items():
        table = biasdb.table("configs")
        entry = table.get(Query().name == name)
        if not entry:
            table.insert({"name": name, "value": value.get("value")})
        else:
            table.update({**value, "value": entry.get("value", value.get("value"))}, Query().name == name)
init()

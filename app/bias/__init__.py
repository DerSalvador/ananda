from functools import cache
import os
import importlib
import inspect
from bias.interface import BiasInterface
from utils import get_logger
from tinydb import TinyDB, Query


logger = get_logger()
CONFIG_PATH = os.getenv("CONFIG_PATH", "/tmp")

biasdb = TinyDB(f"{CONFIG_PATH}/biasdb.json")
# Entries of type {'name': 'CoinGeckoBTC', 'active': True} def init():

def get_biases():
    biases = biasdb.all()
    return biases

def get_bias(bias):
    return biasdb.search(Query().name == bias)

def update_bias(bias, active=True):
    # if doesn't exist, create
    if not biasdb.search(Query().name == bias):
        biasdb.insert({"name": bias, "active": active})
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
    return table.all()

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

@cache
def getInterfaces():
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
                        exising_bias = get_bias(interface_name)
                        if exising_bias:
                            if not exising_bias[0]["active"]:
                                obj.ignore = True
                        if not obj.ignore:
                            interfaces[interface_name] = obj()
                        else:
                            logger.info(f"Ignoring {interface_name}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
    logger.info(f"Getting interfaces: {list(interfaces.keys())}")
    return interfaces

def init():
    for bias in getInterfaces().keys():
        if not biasdb.search(Query().name == bias):
            biasdb.insert({"name": bias, "active": True})
    configs = {
        "GreedAndFearLimit": 10,
        "ReverseTrendCheckBackSeconds": 600,
        "ReverseTrendCheckMinCount": 60,
        "ReverseTrendCheckPercentageAgreeThreshold": 100,
    }
    for name, value in configs.items():
        table = biasdb.table("configs")
        if not table.get(Query().name == name):
            table.insert({"name": name, "value": value})
init()

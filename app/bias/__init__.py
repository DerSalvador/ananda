import os
import importlib
import inspect
from bias.interface import BiasInterface
from utils import get_logger

INTERFACES = {}

logger = get_logger()

# Get the directory of the current file
current_dir = os.path.dirname(__file__)

# List all Python files in the directory
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
                    if not obj.ignore:
                        INTERFACES[interface_name] = obj()
                    else:
                        logger.info(f"Ignoring {interface_name}")
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")

logger.info(f"Loaded interfaces: {list(INTERFACES.keys())}")

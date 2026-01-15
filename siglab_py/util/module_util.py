import os
import glob
import importlib.util
import inspect

def load_module_class(strategy_name: str):
    if not strategy_name:
        return None

    caller_frame = inspect.stack()[1] # [0] = here, [1] = direct caller
    caller_file = caller_frame.filename
    folder = os.path.dirname(caller_file)
    print(f"folder searched: {folder}")

    pattern = os.path.join(folder, "*.py")

    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        print(f"filename: {filename}")
        
        if filename.startswith('_') or filename == 'strategy_executor.py':
            continue

        module_name = filename[:-3]

        try:
            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, strategy_name):
                cls = getattr(module, strategy_name)
                if isinstance(cls, type):
                    print(f"Loaded strategy: {strategy_name} from {filename}")
                    return cls
        except Exception as e:
            print(f"Skipping {filename} – error: {e}")

    print(f"{strategy_name} not found.")
    return None
import glob
import importlib.util

def load_strategy_class(strategy_name: str):
    if not strategy_name:
        return None

    folder = os.path.dirname(os.path.abspath(__file__))
    pattern = os.path.join(folder, "*.py")

    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
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
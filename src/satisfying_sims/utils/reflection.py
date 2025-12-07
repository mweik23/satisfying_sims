def get_class(name: str, module):
    target = name.lower()
    for key, obj in module.__dict__.items():
        if isinstance(obj, type) and key.lower() == target:
            return obj
    raise ValueError(f"Class '{name}' not found in module {module.__name__}")

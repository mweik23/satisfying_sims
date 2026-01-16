# put this near recording.save(...) or inside Recording.save before pickle.dump
import pickle
import types

def find_mappingproxy(obj, path="root", seen=None):
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)

    if isinstance(obj, types.MappingProxyType):
        return path

    # basic containers
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = find_mappingproxy(v, f"{path}[{k!r}]", seen)
            if p: return p
        return None
    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            p = find_mappingproxy(v, f"{path}[{i}]", seen)
            if p: return p
        return None

    # dataclasses / objects
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            p = find_mappingproxy(v, f"{path}.{k}", seen)
            if p: return p

    return None

import pickle
import dataclasses
import types

def _iter_children(obj):
    # dict: check BOTH keys and values
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield ("<key>", k)
            yield (repr(k), v)
        return

    # list/tuple/set
    if isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            yield (f"[{i}]", v)
        return

    # dataclass (works even with slots=True)
    if dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            try:
                yield (f".{f.name}", getattr(obj, f.name))
            except Exception:
                pass
        return

    # normal objects with __dict__
    if hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            yield (f".{k}", v)
        return

    # objects with __slots__
    slots = getattr(type(obj), "__slots__", None)
    if slots:
        if isinstance(slots, str):
            slots = [slots]
        for s in slots:
            try:
                yield (f".{s}", getattr(obj, s))
            except Exception:
                pass
        return

def find_first_unpicklable(obj, path="root", seen=None):
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return None
    seen.add(oid)

    # If this object pickles, stop here.
    try:
        pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        return None
    except Exception as e:
        # If it doesn't, try to descend.
        children = list(_iter_children(obj))
        if not children:
            return (path, obj, e)

        for suffix, child in children:
            res = find_first_unpicklable(child, path + suffix, seen)
            if res is not None:
                return res

        # Couldn't localize further
        return (path, obj, e)

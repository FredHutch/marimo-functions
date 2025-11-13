from pathlib import Path
import json
from typing import Any, Dict, Optional


# Cache app settings in a local folder
class CachedJSON:
    _cache_path: Path
    _cache_fp: str
    default_type: Any

    def __init__(self):
        self._cache_path = Path(self._cache_fp)
        self._cache_path.parent.mkdir(exist_ok=True, parents=True)
        self._cache = None

    def read_cache(self) -> dict:
        """Read the cached settings."""
        if self._cache is None:
            if self._cache_path.exists():
                with open(self._cache_path, "rt") as handle:
                    content = handle.read()
                if len(content) > 1:
                    self._cache = json.loads(content)
                else:
                    self._cache = self.default_type()
            else:
                self._cache = self.default_type()

        return self._cache

    def write_cache(self, obj: Dict[str, Any]):
        with open(self._cache_path, "wt") as handle:
            json.dump(obj, handle, indent=4)
        self._cache = obj


class CachedList(CachedJSON):
    default_type = list


class CachedDict(CachedJSON):
    default_type = dict

    def get(self, kw: str, default: Optional[Any] = None):
        return self.read_cache().get(kw, default)

    def get_safe(self, kw: str, options: list, default: Optional[Any] = None):
        """Only return the value if it is present in the list of provided options."""
        val = self.read_cache().get(kw, default)
        if val in options:
            return val
        else:
            return default

    def get_safe_list(self, kw: str, options: list, default: Optional[Any] = None):
        """Only return the elements of the value if they are present in the list of provided options."""
        val = self.read_cache().get(kw, default)
        val = [v for v in val if v in options]
        if len(val) == 0:
            val = default
        return val

    def set(self, kw: str, val: Any):
        obj = self.read_cache()
        obj[kw] = val
        self.write_cache(obj)


class Settings(CachedDict):
    _cache_fp = ".cache/settings.json"

    def ui(self, kw: str, default: Any):
        return dict(
            value=self.get(kw, default),
            on_change=lambda v: self.set(kw, v)
        )

    def ui_safe(self, kw: str, options: list, default: Any):
        return dict(
            value=self.get_safe(kw, options, default),
            on_change=lambda v: self.set(kw, v)
        )

    def ui_safe_list(self, kw: str, options: list, default: Any):
        return dict(
            value=self.get_safe_list(kw, options, default),
            on_change=lambda v: self.set(kw, v)
        )

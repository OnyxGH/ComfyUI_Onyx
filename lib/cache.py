import itertools
from collections import OrderedDict
from typing import Optional


class Store:
    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


store = Store()


class SimpleLRUCache(OrderedDict):
    def __init__(self, maxsize: int):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)  # mark as recently used
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)  # evict least recently used


class TaggedCache:
    def __init__(self, tag_settings: Optional[dict] = None):
        self._tag_settings = tag_settings or {}  # tag cache size
        self._data = {}

    def __getitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        raise KeyError(f"Key `{key}` does not exist")

    def __setitem__(self, key, value: tuple):
        # value: (tag: str, (islist: bool, data: *))

        # if key already exists, pop old value
        for tag_data in self._data.values():
            if key in tag_data:
                tag_data.pop(key, None)
                break

        tag = value[0]
        if tag not in self._data:

            try:
                from cachetools import LRUCache  # type: ignore

                default_size = 20
                if "ckpt" in tag:
                    default_size = 5
                elif tag in ["latent", "image"]:
                    default_size = 100

                self._data[tag] = LRUCache(maxsize=self._tag_settings.get(tag, default_size))

            except (ImportError, ModuleNotFoundError):
                self._data[tag] = SimpleLRUCache(maxsize=self._tag_settings.get(tag, default_size))
        self._data[tag][key] = value

    def __delitem__(self, key):
        for tag_data in self._data.values():
            if key in tag_data:
                del tag_data[key]
                return
        raise KeyError(f"Key `{key}` does not exist")

    def __contains__(self, key):
        return any(key in tag_data for tag_data in self._data.values())

    def items(self):
        yield from itertools.chain(*map(lambda x: x.items(), self._data.values()))

    def get(self, key, default=None):
        """D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None."""
        for tag_data in self._data.values():
            if key in tag_data:
                return tag_data[key]
        return default

    def clear(self):
        # clear all cache
        self._data = {}


cache_settings = {}
cache = TaggedCache(cache_settings)
cache_count = {}


def update_cache(k, tag, v):
    cache[k] = (tag, v)
    cnt = cache_count.get(k)
    if cnt is None:
        cnt = 0
        cache_count[k] = cnt
    else:
        cache_count[k] += 1


def remove_cache(key):
    global cache
    if key == "*":
        cache = TaggedCache(cache_settings)
    elif key in cache:
        del cache[key]
    else:
        print(f"invalid {key}")

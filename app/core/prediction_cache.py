
from cachetools import TTLCache
from concurrent.futures import Future
from threading import RLock
from copy import deepcopy


class PredictionCache:
    def __init__(self, ttl_seconds=1800, max_entries=256):
        self.cache = TTLCache(
            maxsize=max_entries,
            ttl=ttl_seconds
        )
        self.lock = RLock()
        self.in_flight = {}

    def get_or_compute(self, key, compute_function):
        with self.lock:
            if key in self.cache:
                return deepcopy(self.cache[key]), True

            future = self.in_flight.get(key)

            if future is None:
                future = Future()
                self.in_flight[key] = future
                is_owner = True
            else:
                is_owner = False

        if not is_owner:
            return deepcopy(future.result()), True

        try:
            result = compute_function()

            with self.lock:
                self.cache[key] = deepcopy(result)

            future.set_result(result)
            return deepcopy(result), False

        except Exception as error:
            future.set_exception(error)
            raise

        finally:
            with self.lock:
                self.in_flight.pop(key, None)


forecast_cache = PredictionCache(
    ttl_seconds=1800,
    max_entries=256
)

general_forecast_cache = PredictionCache(
    ttl_seconds=1800,
    max_entries=64
)

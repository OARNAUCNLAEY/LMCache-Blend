# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import OrderedDict
from typing import Any

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.cache_policy.base_policy import BaseCachePolicy

from lmcache.observability import LMCStatsMonitor, JsonLogger, CacheEvent
logger = init_logger(__name__)


class LRUCachePolicy(BaseCachePolicy[OrderedDict[CacheEngineKey, Any]]):
    """
    LRU cache policy.
    """

    def __init__(self):
        logger.info("Initializing LRUCachePolicy")
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

    def init_mutable_mapping(self) -> OrderedDict[CacheEngineKey, Any]:
        return OrderedDict()

    def update_on_hit(
        self,
        key: CacheEngineKey,
        cache_dict: OrderedDict[CacheEngineKey, Any],
    ) -> None:
        self.stats_monitor.add_cache_event(CacheEvent.HIT, key)
        cache_dict.move_to_end(key)

    def update_on_put(
        self,
        key: CacheEngineKey,
    ) -> None:
        # No action needed for LRU on put, as the key is already at the end.
        self.stats_monitor.add_cache_event(CacheEvent.STORE, key)
        return

    def update_on_force_evict(
        self,
        key: CacheEngineKey,
    ) -> None:
        
        self.stats_monitor.add_cache_event(CacheEvent.EVICT, key)
        return 

    # NOTE(Jiayi): We do best effort to get eviction candidates so the number
    # of returned keys mignt be smaller than num_candidates.
    def get_evict_candidates(
        self,
        cache_dict: OrderedDict[CacheEngineKey, Any],
        num_candidates: int = 1,
    ) -> list[CacheEngineKey]:
        evict_keys = []
        for key, cache in cache_dict.items():
            if not cache.can_evict:
                continue
            evict_keys.append(key)
            if len(evict_keys) == num_candidates:
                break

        return evict_keys

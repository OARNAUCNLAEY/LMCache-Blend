# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import os
import threading
import time

# Third Party
from prometheus_client import REGISTRY
import prometheus_client

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import thread_safe, CacheEngineKey
from enum import Enum
from datetime import datetime
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
logger = init_logger(__name__)
class CacheEvent(Enum):
    STORE = 1
    EVICT = 2
    HIT = 3
    BLOCKING_HIT = 4
    NON_BLOCKING_HIT = 5

@dataclass
class LMCacheStats:
    # Counter (Note that these are incremental values,
    # which will accumulate over time in Counter)
    interval_retrieve_requests: int
    interval_store_requests: int
    interval_lookup_requests: int
    interval_requested_tokens: int
    interval_hit_tokens: int
    interval_lookup_tokens: int
    interval_lookup_hits: int

    interval_remote_read_requests: int
    interval_remote_read_bytes: int
    interval_remote_write_requests: int
    interval_remote_write_bytes: int

    interval_remote_time_to_get: List[float]
    interval_remote_time_to_put: List[float]
    interval_remote_time_to_get_sync: List[float]

    interval_remote_ping_latency: float  # Ping latency in milliseconds
    interval_remote_ping_errors: int  # Number of ping errors
    interval_remote_ping_success: int  # Number of ping successes
    interval_remote_ping_error_code: int  # Latest ping error code

    interval_local_cpu_evict_count: int  # evict count
    interval_local_cpu_evict_keys_count: int  # evict keys count
    interval_local_cpu_evict_failed_count: int  # evict failed count

    # Real time value measurements (will be reset after each log)
    retrieve_hit_rate: float
    lookup_hit_rate: float

    local_cache_usage_bytes: int  # Size of the used local cache in bytes
    remote_cache_usage_bytes: int  # Size of the used remote cache in bytes
    local_storage_usage_bytes: int  # Size of the used local storage in bytes

    active_memory_objs_count: int  # the number of active memory objects
    pinned_memory_objs_count: int  # the number of pinned memory objects

    # Distribution measurements
    time_to_retrieve: List[float]
    time_to_store: List[float]
    retrieve_speed: List[float]  # Tokens per second
    store_speed: List[float]  # Tokens per second

    cache_events: List
    hash_chunk_mapping: Dict


@dataclass
class LookupRequestStats:
    num_tokens: int
    hit_tokens: int


@dataclass
class RetrieveRequestStats:
    num_tokens: int
    local_hit_tokens: int
    remote_hit_tokens: int  # Not used for now
    start_time: float
    end_time: float

    def time_to_retrieve(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def retrieve_speed(self):
        if self.time_to_retrieve() == 0:
            return 0
        return (
            self.local_hit_tokens + self.remote_hit_tokens
        ) / self.time_to_retrieve()


@dataclass
class StoreRequestStats:
    num_tokens: int
    start_time: float
    end_time: float

    def time_to_store(self):
        if self.end_time == 0:
            return 0
        return self.end_time - self.start_time

    def store_speed(self):
        if self.time_to_store() == 0:
            return 0
        return self.num_tokens / self.time_to_store()


class LMCStatsMonitor:
    def __init__(self):
        # Interval metrics that will be reset after each log
        # Accumulate incremental values in the Prometheus Counter
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0
        self.interval_lookup_requests = 0
        self.interval_requested_tokens = 0  # total requested tokens retrieve
        self.interval_hit_tokens = 0  # total hit tokens retrieve
        self.interval_lookup_tokens = 0  # total requested tokens lookup
        self.interval_lookup_hits = 0  # total hit tokens lookup

        # remote backends read/write metrics
        self.interval_remote_read_requests = 0
        self.interval_remote_read_bytes = 0
        self.interval_remote_write_requests = 0
        self.interval_remote_write_bytes = 0

        # remote backends get/put cost time metrics
        self.interval_remote_time_to_get: List[float] = []
        self.interval_remote_time_to_put: List[float] = []
        # the time of get value from remote backends synchronously,
        # which includes rpc and schedule time
        self.interval_remote_time_to_get_sync: List[float] = []

        self.interval_remote_ping_latency = 0
        self.interval_remote_ping_errors = 0
        self.interval_remote_ping_success = 0
        self.interval_remote_ping_error_code = 0  # 0 means success

        self.interval_local_cpu_evict_count = 0
        self.interval_local_cpu_evict_keys_count = 0
        self.interval_local_cpu_evict_failed_count = 0

        self.local_cache_usage_bytes = 0
        self.remote_cache_usage_bytes = 0
        self.local_storage_usage_bytes = 0

        self.active_memory_objs_count = 0
        self.pinned_memory_objs_count = 0

        self.retrieve_requests: Dict[int, RetrieveRequestStats] = {}
        self.store_requests: Dict[int, StoreRequestStats] = {}

        self.retrieve_request_id = 0
        self.store_request_id = 0

        self.hash_chunk_mapping = {-1 : "Hash Not Found"}
        self.cache_events = []
        self.enable_logging = 0
        if "LMCACHE_ENABLE_CACHE_LOGGING" in os.environ and int(os.environ["LMCACHE_ENABLE_CACHE_LOGGING"]) == 1:
            self.enable_logging = 1
        self.model_name = os.environ["MODEL"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
        )

    @thread_safe
    def on_lookup_request(self, num_tokens: int):
        """
        This function is called when a lookup request is sent to the cache.
        It will record the number of tokens requested.
        """
        self.interval_lookup_requests += 1
        self.interval_lookup_tokens += num_tokens

    @thread_safe
    def on_lookup_finished(self, num_hit_tokens: int):
        """
        This function is called when a lookup request is finished.
        It will record the number of tokens hit.
        """
        self.interval_lookup_hits += num_hit_tokens

    @thread_safe
    def on_retrieve_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in
        on_retrieve_finished
        """
        curr_time = time.time()
        retrieve_stats = RetrieveRequestStats(
            num_tokens=num_tokens,
            local_hit_tokens=0,
            remote_hit_tokens=0,
            start_time=curr_time,
            end_time=0,
        )
        self.interval_requested_tokens += num_tokens
        self.interval_retrieve_requests += 1
        self.retrieve_requests[self.retrieve_request_id] = retrieve_stats
        self.retrieve_request_id += 1
        return self.retrieve_request_id - 1

    @thread_safe
    def on_retrieve_finished(self, request_id: int, retrieved_tokens: int):
        curr_time = time.time()
        assert request_id in self.retrieve_requests
        retrieve_stats = self.retrieve_requests[request_id]
        retrieve_stats.local_hit_tokens = retrieved_tokens
        retrieve_stats.end_time = curr_time
        self.interval_hit_tokens += retrieved_tokens

    @thread_safe
    def on_store_request(self, num_tokens: int) -> int:
        """
        Returns the internal "request id" that will be used in on_store_finished
        """
        curr_time = time.time()
        store_stats = StoreRequestStats(
            num_tokens=num_tokens, start_time=curr_time, end_time=0
        )
        self.interval_store_requests += 1
        self.store_requests[self.store_request_id] = store_stats
        self.store_request_id += 1
        return self.store_request_id - 1

    @thread_safe
    def on_store_finished(self, request_id: int, num_tokens: int = -1):
        curr_time = time.time()
        assert request_id in self.store_requests
        store_stats = self.store_requests[request_id]
        store_stats.end_time = curr_time
        if num_tokens >= 0:
            store_stats.num_tokens = num_tokens

    @thread_safe
    def add_to_hash_mapping(self, key : CacheEngineKey, token_id: List[int]):
        if self.enable_logging == 1:
            self.hash_chunk_mapping[key.chunk_hash] = self.tokenizer.decode(token_id)
        return
    
    @thread_safe
    def add_cache_event(self, event: CacheEvent, key : CacheEngineKey):
        if self.enable_logging == 1:
            self.cache_events.append((event.name, key.chunk_hash, str(key)))
        return

    @thread_safe
    def update_local_cache_usage(self, usage: int):
        self.local_cache_usage_bytes = usage

    @thread_safe
    def update_remote_cache_usage(self, usage: int):
        self.remote_cache_usage_bytes = usage

    @thread_safe
    def update_local_storage_usage(self, usage: int):
        self.local_storage_usage_bytes = usage

    @thread_safe
    def update_interval_remote_read_metrics(self, read_bytes: int):
        self.interval_remote_read_requests += 1
        self.interval_remote_read_bytes += read_bytes

    @thread_safe
    def update_interval_remote_write_metrics(self, write_bytes: int):
        self.interval_remote_write_requests += 1
        self.interval_remote_write_bytes += write_bytes

    @thread_safe
    def update_interval_remote_time_to_get(self, get_time: float):
        self.interval_remote_time_to_get.append(get_time)

    @thread_safe
    def update_interval_remote_time_to_put(self, put_time: float):
        self.interval_remote_time_to_put.append(put_time)

    @thread_safe
    def update_interval_remote_time_to_get_sync(self, get_time_sync: float):
        self.interval_remote_time_to_get_sync.append(get_time_sync)

    @thread_safe
    def update_remote_ping_latency(self, latency: float):
        self.interval_remote_ping_latency = latency

    @thread_safe
    def update_remote_ping_error_code(self, error_code: int):
        """Update ping error code"""
        self.interval_remote_ping_error_code = error_code
        if error_code != 0:
            self.interval_remote_ping_errors += 1
        else:
            self.interval_remote_ping_success += 1

    @thread_safe
    def update_local_cpu_evict_metrics(self, evict_keys_count: int):
        self.interval_local_cpu_evict_count += 1
        self.interval_local_cpu_evict_keys_count += evict_keys_count

    @thread_safe
    def update_local_cpu_evict_failed_count(self, evict_failed_count: int):
        self.interval_local_cpu_evict_failed_count += evict_failed_count

    @thread_safe
    def update_active_memory_objs_count(self, active_memory_objs_count: int):
        self.active_memory_objs_count = active_memory_objs_count

    @thread_safe
    def update_pinned_memory_objs_count(self, delta: int):
        self.pinned_memory_objs_count += delta

    def _clear(self):
        """
        Clear all the distribution stats
        """
        self.interval_retrieve_requests = 0
        self.interval_store_requests = 0
        self.interval_lookup_requests = 0

        self.interval_requested_tokens = 0
        self.interval_hit_tokens = 0
        self.interval_lookup_tokens = 0
        self.interval_lookup_hits = 0

        self.interval_remote_read_requests = 0
        self.interval_remote_read_bytes = 0
        self.interval_remote_write_requests = 0
        self.interval_remote_write_bytes = 0

        self.interval_remote_time_to_get.clear()
        self.interval_remote_time_to_put.clear()
        self.interval_remote_time_to_get_sync.clear()

        self.interval_remote_ping_latency = 0
        self.interval_remote_ping_errors = 0
        self.interval_remote_ping_success = 0
        self.interval_remote_ping_error_code = 0

        self.interval_local_cpu_evict_count = 0
        self.interval_local_cpu_evict_keys_count = 0
        self.interval_local_cpu_evict_failed_count = 0

        new_retrieve_requests = {}
        for request_id, retrieve_stats in self.retrieve_requests.items():
            if retrieve_stats.end_time == 0:
                new_retrieve_requests[request_id] = retrieve_stats
        self.retrieve_requests = new_retrieve_requests

        new_store_requests = {}
        for request_id, store_stats in self.store_requests.items():
            if store_stats.end_time == 0:
                new_store_requests[request_id] = store_stats
        self.store_requests = new_store_requests
    
        self.cache_events = []
        self.hash_chunk_mapping = {-1 : "Hash Not Found"}
        

    @thread_safe
    def get_stats_and_clear(self) -> LMCacheStats:
        """
        This function should be called with by prometheus adapter with
        a specific interval.
        The function will return the latest states between the current
        call and the previous call.
        """
        retrieve_hit_rate = (
            0
            if self.interval_requested_tokens == 0
            else self.interval_hit_tokens / self.interval_requested_tokens
        )

        lookup_hit_rate = (
            0
            if self.interval_lookup_tokens == 0
            else self.interval_lookup_hits / self.interval_lookup_tokens
        )

        def filter_out_invalid(stats: List[float]):
            return [x for x in stats if x != 0]

        time_to_retrieve = filter_out_invalid(
            [stats.time_to_retrieve() for stats in self.retrieve_requests.values()]
        )

        time_to_store = filter_out_invalid(
            [stats.time_to_store() for stats in self.store_requests.values()]
        )

        retrieve_speed = filter_out_invalid(
            [stats.retrieve_speed() for stats in self.retrieve_requests.values()]
        )

        store_speed = filter_out_invalid(
            [stats.store_speed() for stats in self.store_requests.values()]
        )

        ret = LMCacheStats(
            interval_retrieve_requests=self.interval_retrieve_requests,
            interval_store_requests=self.interval_store_requests,
            interval_lookup_requests=self.interval_lookup_requests,
            interval_requested_tokens=self.interval_requested_tokens,
            interval_hit_tokens=self.interval_hit_tokens,
            interval_lookup_tokens=self.interval_lookup_tokens,
            interval_lookup_hits=self.interval_lookup_hits,
            interval_remote_read_requests=self.interval_remote_read_requests,
            interval_remote_read_bytes=self.interval_remote_read_bytes,
            interval_remote_write_requests=self.interval_remote_write_requests,
            interval_remote_write_bytes=self.interval_remote_write_bytes,
            interval_remote_time_to_get=self.interval_remote_time_to_get.copy(),
            interval_remote_time_to_put=self.interval_remote_time_to_put.copy(),
            interval_remote_time_to_get_sync=self.interval_remote_time_to_get_sync.copy(),
            interval_remote_ping_latency=self.interval_remote_ping_latency,
            interval_remote_ping_errors=self.interval_remote_ping_errors,
            interval_remote_ping_success=self.interval_remote_ping_success,
            interval_remote_ping_error_code=self.interval_remote_ping_error_code,
            retrieve_hit_rate=retrieve_hit_rate,
            lookup_hit_rate=lookup_hit_rate,
            interval_local_cpu_evict_count=self.interval_local_cpu_evict_count,
            interval_local_cpu_evict_keys_count=self.interval_local_cpu_evict_keys_count,
            interval_local_cpu_evict_failed_count=self.interval_local_cpu_evict_failed_count,
            local_cache_usage_bytes=self.local_cache_usage_bytes,
            remote_cache_usage_bytes=self.remote_cache_usage_bytes,
            local_storage_usage_bytes=self.local_storage_usage_bytes,
            active_memory_objs_count=self.active_memory_objs_count,
            pinned_memory_objs_count=self.pinned_memory_objs_count,
            time_to_retrieve=time_to_retrieve,
            time_to_store=time_to_store,
            retrieve_speed=retrieve_speed,
            store_speed=store_speed,
            cache_events = self.cache_events,
            hash_chunk_mapping = self.hash_chunk_mapping,
        )
        self._clear()
        return ret

    _instance = None

    @staticmethod
    def GetOrCreate() -> "LMCStatsMonitor":
        if LMCStatsMonitor._instance is None:
            LMCStatsMonitor._instance = LMCStatsMonitor()
        return LMCStatsMonitor._instance

    @staticmethod
    def DestroyInstance():
        LMCStatsMonitor._instance = None

    @staticmethod
    def unregister_all_metrics():
        collectors = list(REGISTRY._collector_to_names.keys())
        for collector in collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass


class PrometheusLogger:
    _gauge_cls = prometheus_client.Gauge
    _counter_cls = prometheus_client.Counter
    _histogram_cls = prometheus_client.Histogram

    def __init__(self, metadata: LMCacheEngineMetadata):
        # Ensure PROMETHEUS_MULTIPROC_DIR is set before any metric registration
        if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
            default_dir = "/tmp/lmcache_prometheus"
            os.environ["PROMETHEUS_MULTIPROC_DIR"] = default_dir
            if not os.path.exists(default_dir):
                os.makedirs(default_dir, exist_ok=True)

        self.metadata = metadata

        self.labels = self._metadata_to_labels(metadata)
        labelnames = list(self.labels.keys())

        self.counter_num_retrieve_requests = self._counter_cls(
            name="lmcache:num_retrieve_requests",
            documentation="Total number of retrieve requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_store_requests = self._counter_cls(
            name="lmcache:num_store_requests",
            documentation="Total number of store requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_lookup_requests = self._counter_cls(
            name="lmcache:num_lookup_requests",
            documentation="Total number of lookup requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_requested_tokens = self._counter_cls(
            name="lmcache:num_requested_tokens",
            documentation="Total number of tokens requested from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_hit_tokens = self._counter_cls(
            name="lmcache:num_hit_tokens",
            documentation="Total number of tokens hit in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_lookup_tokens = self._counter_cls(
            name="lmcache:num_lookup_tokens",
            documentation="Total number of tokens requested in lookup from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_lookup_hits = self._counter_cls(
            name="lmcache:num_lookup_hits",
            documentation="Total number of tokens hit in lookup from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_read_requests = self._counter_cls(
            name="lmcache:num_remote_read_requests",
            documentation="Total number of requests read from "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_read_bytes = self._counter_cls(
            name="lmcache:num_remote_read_bytes",
            documentation="Total number of bytes read from remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_requests = self._counter_cls(
            name="lmcache:num_remote_write_requests",
            documentation="Total number of requests write to "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_bytes = self._counter_cls(
            name="lmcache:num_remote_write_bytes",
            documentation="Total number of bytes write to remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_local_cpu_evict_count = self._counter_cls(
            name="lmcache:local_cpu_evict_count",
            documentation="Total number of evict in local cpu backend",
            labelnames=labelnames,
        )

        self.counter_local_cpu_evict_keys_count = self._counter_cls(
            name="lmcache:local_cpu_evict_keys_count",
            documentation="Total number of evict keys in local cpu backend",
            labelnames=labelnames,
        )

        self.counter_local_cpu_evict_failed_count = self._counter_cls(
            name="lmcache:local_cpu_evict_failed_count",
            documentation="Total number of failed eviction in local cpu backend",
            labelnames=labelnames,
        )

        self.gauge_retrieve_hit_rate = self._gauge_cls(
            name="lmcache:retrieve_hit_rate",
            documentation="Hit rate of lmcache retrieve requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        self.gauge_lookup_hit_rate = self._gauge_cls(
            name="lmcache:lookup_hit_rate",
            documentation="Hit rate of lmcache lookup requests since last log",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )

        self.gauge_local_cache_usage = self._gauge_cls(
            name="lmcache:local_cache_usage",
            documentation="Local cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_remote_cache_usage = self._gauge_cls(
            name="lmcache:remote_cache_usage",
            documentation="Remote cache usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_local_storage_usage = self._gauge_cls(
            name="lmcache:local_storage_usage",
            documentation="Local storage usage (bytes) of lmcache",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_active_memory_objs_count = self._gauge_cls(
            name="lmcache:active_memory_objs_count",
            documentation="The number of active memory objects",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        self.gauge_pinned_memory_objs_count = self._gauge_cls(
            name="lmcache:pinned_memory_objs_count",
            documentation="The number of pinned memory objects",
            labelnames=labelnames,
            multiprocess_mode="sum",
        )

        time_to_retrieve_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_retrieve = self._histogram_cls(
            name="lmcache:time_to_retrieve",
            documentation="Time to retrieve from lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_retrieve_buckets,
        )

        time_to_store_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_store = self._histogram_cls(
            name="lmcache:time_to_store",
            documentation="Time to store to lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_store_buckets,
        )

        retrieve_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_retrieve_speed = self._histogram_cls(
            name="lmcache:retrieve_speed",
            documentation="Retrieve speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=retrieve_speed_buckets,
        )

        store_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_store_speed = self._histogram_cls(
            name="lmcache:store_speed",
            documentation="Store speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=store_speed_buckets,
        )

        remote_time_to_get = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get = self._histogram_cls(
            name="lmcache:remote_time_to_get",
            documentation="Time to get from remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get,
        )

        remote_time_to_put = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_put = self._histogram_cls(
            name="lmcache:remote_time_to_put",
            documentation="Time to put to remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_put,
        )

        remote_time_to_get_sync = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get_sync = self._histogram_cls(
            name="lmcache:remote_time_to_get_sync",
            documentation="Time to get from remote backends synchronously(ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get_sync,
        )

        # Ping latency metrics: use a gauge to record the latest ping latency
        self.gauge_remote_ping_latency = self._gauge_cls(
            name="lmcache:remote_ping_latency",
            documentation="Latest ping latency to remote backends (ms)",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self.counter_remote_ping_errors = self._counter_cls(
            name="lmcache:remote_ping_errors",
            documentation="Number of ping errors to remote backends",
            labelnames=labelnames,
        )
        self.counter_remote_ping_successes = self._counter_cls(
            name="lmcache:remote_ping_successes",
            documentation="Number of ping successes to remote backends",
            labelnames=labelnames,
        )
        self.gauge_remote_ping_error_code = self._gauge_cls(
            name="lmcache:remote_ping_error_code",
            documentation="Latest ping error code to remote backends",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        )
        self._dynamic_metrics(labelnames)

    def _dynamic_metrics(self, labelnames):
        """
        Dynamically get value by lambda function while capture
        """
        self.local_cpu_hot_cache_count = self._gauge_cls(
            name="lmcache:local_cpu_hot_cache_count",
            documentation="The size of the hot_cache",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)
        self.local_cpu_keys_in_request_count = self._gauge_cls(
            name="lmcache:local_cpu_keys_in_request_count",
            documentation="The size of the keys_in_request",
            labelnames=labelnames,
            multiprocess_mode="livemostrecent",
        ).labels(**self.labels)

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.labels(**self.labels).set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        if data < 0:
            return
        counter.labels(**self.labels).inc(data)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.labels(**self.labels).observe(value)

    def log_prometheus(self, stats: LMCacheStats):
        self._log_counter(
            self.counter_num_retrieve_requests, stats.interval_retrieve_requests
        )
        self._log_counter(
            self.counter_num_store_requests, stats.interval_store_requests
        )
        self._log_counter(
            self.counter_num_lookup_requests, stats.interval_lookup_requests
        )

        self._log_counter(
            self.counter_num_requested_tokens, stats.interval_requested_tokens
        )
        self._log_counter(self.counter_num_hit_tokens, stats.interval_hit_tokens)
        self._log_counter(self.counter_num_lookup_tokens, stats.interval_lookup_tokens)
        self._log_counter(self.counter_num_lookup_hits, stats.interval_lookup_hits)

        self._log_counter(
            self.counter_num_remote_read_requests,
            stats.interval_remote_read_requests,
        )
        self._log_counter(
            self.counter_num_remote_read_bytes, stats.interval_remote_read_bytes
        )
        self._log_counter(
            self.counter_num_remote_write_requests,
            stats.interval_remote_write_requests,
        )
        self._log_counter(
            self.counter_num_remote_write_bytes,
            stats.interval_remote_write_bytes,
        )
        self._log_counter(
            self.counter_local_cpu_evict_count,
            stats.interval_local_cpu_evict_count,
        )
        self._log_counter(
            self.counter_local_cpu_evict_keys_count,
            stats.interval_local_cpu_evict_keys_count,
        )
        self._log_counter(
            self.counter_local_cpu_evict_failed_count,
            stats.interval_local_cpu_evict_failed_count,
        )

        self._log_gauge(self.gauge_retrieve_hit_rate, stats.retrieve_hit_rate)

        self._log_gauge(self.gauge_lookup_hit_rate, stats.lookup_hit_rate)

        self._log_gauge(self.gauge_local_cache_usage, stats.local_cache_usage_bytes)

        self._log_gauge(self.gauge_remote_cache_usage, stats.remote_cache_usage_bytes)

        self._log_gauge(self.gauge_local_storage_usage, stats.local_storage_usage_bytes)

        self._log_histogram(self.histogram_time_to_retrieve, stats.time_to_retrieve)

        self._log_histogram(self.histogram_time_to_store, stats.time_to_store)

        self._log_histogram(self.histogram_retrieve_speed, stats.retrieve_speed)

        self._log_histogram(self.histogram_store_speed, stats.store_speed)

        self._log_histogram(
            self.histogram_remote_time_to_get, stats.interval_remote_time_to_get
        )
        self._log_histogram(
            self.histogram_remote_time_to_put, stats.interval_remote_time_to_put
        )
        self._log_histogram(
            self.histogram_remote_time_to_get_sync,
            stats.interval_remote_time_to_get_sync,
        )
        self._log_gauge(
            self.gauge_remote_ping_latency, stats.interval_remote_ping_latency
        )
        self._log_counter(
            self.counter_remote_ping_errors, stats.interval_remote_ping_errors
        )
        self._log_counter(
            self.counter_remote_ping_successes, stats.interval_remote_ping_success
        )
        self._log_gauge(
            self.gauge_remote_ping_error_code, stats.interval_remote_ping_error_code
        )
        self._log_gauge(
            self.gauge_active_memory_objs_count, stats.active_memory_objs_count
        )
        self._log_gauge(
            self.gauge_pinned_memory_objs_count, stats.pinned_memory_objs_count
        )

    @staticmethod
    def _metadata_to_labels(metadata: LMCacheEngineMetadata):
        return {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id,
        }

    _instance = None

    @staticmethod
    def GetOrCreate(metadata: LMCacheEngineMetadata) -> "PrometheusLogger":
        if PrometheusLogger._instance is None:
            PrometheusLogger._instance = PrometheusLogger(metadata)
        # assert PrometheusLogger._instance.metadata == metadata, \
        #    "PrometheusLogger instance already created with different metadata"
        if PrometheusLogger._instance.metadata != metadata:
            logger.error(
                "PrometheusLogger instance already created with"
                "different metadata. This should not happen except "
                "in test"
            )
        return PrometheusLogger._instance

    @staticmethod
    def GetInstance() -> "PrometheusLogger":
        assert PrometheusLogger._instance is not None, (
            "PrometheusLogger instance not created yet"
        )
        return PrometheusLogger._instance
    
class CounterJson:
    
    def __init__(self, name,  documentation, labelnames):
        self.name = name
        self.documentation=documentation
        self.labelnames=labelnames
        self.counter = {}
        for label in labelnames:
            self.counter[label] = 0
    
    def inc(self, data):
        for label in self.labelnames:
            if isinstance(data, (float, int)):
                self.counter[label] += data
            else:
                self.counter[label] += data.item()
    
    def __str__(self):
        return {"Name" : self.name, "Documentation": self.documentation, "Counter": self.counter}
    
    def get_dict(self):
        return {"Name" : self.name, "Documentation": self.documentation, "Counter": self.counter}

class GaugeJson:
    
    def __init__(self, name,  documentation, labelnames):
        self.name = name
        self.documentation=documentation
        self.labelnames=labelnames
        self.gauge = {}
        for label in labelnames:
            self.gauge[label] = []
    def set(self, data):
        current_utc_time = str(datetime.now())
        for label in self.labelnames:
            if isinstance(data, (float, int)):
                self.gauge[label].append((current_utc_time, data))
            else:
                self.gauge[label].append((current_utc_time, data.item()))
    def __str__(self):
        return {"Name" : self.name, "Documentation": self.documentation, "Gauge": self.gauge}
    
    def get_dict(self):
        return {"Name" : self.name, "Documentation": self.documentation, "Gauge": self.gauge}
    
class HistogramJson:

    def __init__(self, name, documentation, labelnames, buckets):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self.buckets = buckets
        self.histogram = {}
        for label in self.labelnames:
            self.histogram[label] = {"buckets" : self.buckets + ["inf"], "frequency": [0]*(len(self.buckets) + 1)}

    def observe(self, data):
        ind = self.find_index(data)
        for label in self.labelnames:
            self.histogram[label]["frequency"][ind] += 1

    def find_index(self, data):
        for i in range(len(self.buckets) - 1):
            if self.buckets[i] <= data < self.buckets[i + 1]:
                return i
        return len(self.buckets)
    
    def __str__(self):
        return {"Name" : self.name, "Documentation": self.documentation, "histogram": self.histogram}
    
    def get_dict(self):
        return {"Name" : self.name, "Documentation": self.documentation, "histogram": self.histogram}


class JsonLogger:
    _gauge_cls = GaugeJson
    _counter_cls = CounterJson
    _histogram_cls = HistogramJson
    _hash_token_mapping = {}
    _cache_events = []
    _cache_event_counter_by_hash = {}

    def _commit_json(self):
        sorted_events = {}
        for event in CacheEvent._member_names_:
            sorted_events[event] = []
            for hash_key in self._cache_event_counter_by_hash:
                if event not in self._cache_event_counter_by_hash[hash_key]:
                    break
                sorted_events[event].append((self._hash_token_mapping[hash_key], self._cache_event_counter_by_hash[hash_key][event]))
            sorted_events[event] = sorted(sorted_events[event], key=lambda x: -1*x[1])
        custom_cache_event_counter = {}
        for key in self._cache_event_counter_by_hash:
            custom_cache_event_counter[self._hash_token_mapping[key]] = self._cache_event_counter_by_hash[key]
        final_dict = {
            "labels": self.labels,
            self.counter_num_retrieve_requests.name : self.counter_num_retrieve_requests.get_dict(),
            self.counter_num_store_requests.name : self.counter_num_store_requests.get_dict(),
            self.counter_num_requested_tokens.name : self.counter_num_requested_tokens.get_dict(),
            self.counter_num_hit_tokens.name : self.counter_num_hit_tokens.get_dict(),
            self.counter_num_remote_read_requests.name : self.counter_num_remote_read_requests.get_dict(),
            self.counter_num_remote_read_bytes.name : self.counter_num_remote_read_bytes.get_dict(),
            self.counter_num_remote_write_requests.name : self.counter_num_remote_write_requests.get_dict(),
            self.counter_num_remote_write_bytes.name :  self.counter_num_remote_write_bytes.get_dict(),
            self.gauge_cache_hit_rate.name : self.gauge_cache_hit_rate.get_dict(),
            self.gauge_local_cache_usage.name : self.gauge_local_cache_usage.get_dict(),
            self.gauge_remote_cache_usage.name : self.gauge_remote_cache_usage.get_dict(),
            self.gauge_local_storage_usage.name : self.gauge_local_storage_usage.get_dict(),
            self.histogram_time_to_retrieve.name : self.histogram_time_to_retrieve.get_dict(),
            self.histogram_time_to_store.name : self.histogram_time_to_store.get_dict(),
            self.histogram_retrieve_speed.name : self.histogram_retrieve_speed.get_dict(),
            self.histogram_store_speed.name : self.histogram_store_speed.get_dict(),
            self.histogram_remote_time_to_get.name : self.histogram_remote_time_to_get.get_dict(),
            self.histogram_remote_time_to_put.name : self.histogram_remote_time_to_put.get_dict(),
            self.histogram_remote_time_to_get_sync.name : self.histogram_remote_time_to_get_sync.get_dict(),
            self.gauge_remote_ping_latency.name : self.gauge_remote_ping_latency.get_dict(),
            self.counter_remote_ping_errors.name : self.counter_remote_ping_errors.get_dict(),
            self.counter_remote_ping_successes.name : self.counter_remote_ping_successes.get_dict(),
            self.gauge_remote_ping_error_code.name : self.gauge_remote_ping_error_code.get_dict(),
            "Top chunks by event": sorted_events,
            "Cache Events by strings": custom_cache_event_counter,
            "Hash <-> token" : self._hash_token_mapping,
            "Cache Events" : [(event, hash_chunk, key) for (event, hash_chunk, key) in self._cache_events],
        }
        import json
        with open(self.path, 'w') as json_file:
            json.dump(final_dict, json_file, indent=4)
    
    def __init__(self, metadata: LMCacheEngineMetadata):

        self.path = "/storage/containers/logs/lmcache/metrics_" + str(datetime.now()) + ".json"
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.metadata = metadata

        self.labels = self._metadata_to_labels(metadata)
        labelnames = list(self.labels.keys())

        self.counter_num_retrieve_requests = self._counter_cls(
            name="lmcache:num_retrieve_requests",
            documentation="Total number of retrieve requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_store_requests = self._counter_cls(
            name="lmcache:num_store_requests",
            documentation="Total number of store requests sent to lmcache",
            labelnames=labelnames,
        )

        self.counter_num_requested_tokens = self._counter_cls(
            name="lmcache:num_requested_tokens",
            documentation="Total number of tokens requested from lmcache",
            labelnames=labelnames,
        )

        self.counter_num_hit_tokens = self._counter_cls(
            name="lmcache:num_hit_tokens",
            documentation="Total number of tokens hit in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_read_requests = self._counter_cls(
            name="lmcache:num_remote_read_requests",
            documentation="Total number of requests read from "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_read_bytes = self._counter_cls(
            name="lmcache:num_remote_read_bytes",
            documentation="Total number of bytes read from remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_requests = self._counter_cls(
            name="lmcache:num_remote_write_requests",
            documentation="Total number of requests write to "
            "remote backends in lmcache",
            labelnames=labelnames,
        )

        self.counter_num_remote_write_bytes = self._counter_cls(
            name="lmcache:num_remote_write_bytes",
            documentation="Total number of bytes write to remote backends in lmcache",
            labelnames=labelnames,
        )

        self.gauge_cache_hit_rate = self._gauge_cls(
            name="lmcache:cache_hit_rate",
            documentation="Cache hit rate of lmcache since last log",
            labelnames=labelnames,
        )

        self.gauge_local_cache_usage = self._gauge_cls(
            name="lmcache:local_cache_usage",
            documentation="Local cache usage (bytes) of lmcache",
            labelnames=labelnames,
        )

        self.gauge_remote_cache_usage = self._gauge_cls(
            name="lmcache:remote_cache_usage",
            documentation="Remote cache usage (bytes) of lmcache",
            labelnames=labelnames,
        )

        self.gauge_local_storage_usage = self._gauge_cls(
            name="lmcache:local_storage_usage",
            documentation="Local storage usage (bytes) of lmcache",
            labelnames=labelnames,
        )

        time_to_retrieve_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_retrieve = self._histogram_cls(
            name="lmcache:time_to_retrieve",
            documentation="Time to retrieve from lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_retrieve_buckets,
        )

        time_to_store_buckets = [
            0.001,
            0.005,
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.1,
            0.25,
            0.5,
            0.75,
            1.0,
            2.5,
            5.0,
            7.5,
            10.0,
        ]
        self.histogram_time_to_store = self._histogram_cls(
            name="lmcache:time_to_store",
            documentation="Time to store to lmcache (seconds)",
            labelnames=labelnames,
            buckets=time_to_store_buckets,
        )

        retrieve_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_retrieve_speed = self._histogram_cls(
            name="lmcache:retrieve_speed",
            documentation="Retrieve speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=retrieve_speed_buckets,
        )

        store_speed_buckets = [
            1,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
        ]
        self.histogram_store_speed = self._histogram_cls(
            name="lmcache:store_speed",
            documentation="Store speed of lmcache (tokens per second)",
            labelnames=labelnames,
            buckets=store_speed_buckets,
        )

        remote_time_to_get = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get = self._histogram_cls(
            name="lmcache:remote_time_to_get",
            documentation="Time to get from remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get,
        )

        remote_time_to_put = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_put = self._histogram_cls(
            name="lmcache:remote_time_to_put",
            documentation="Time to put to remote backends (ms)",
            labelnames=labelnames,
            buckets=remote_time_to_put,
        )

        remote_time_to_get_sync = [
            1,
            5,
            10,
            20,
            40,
            60,
            80,
            100,
            250,
            500,
            750,
            1000,
            2500,
            5000,
            7500,
            10000,
        ]
        self.histogram_remote_time_to_get_sync = self._histogram_cls(
            name="lmcache:remote_time_to_get_sync",
            documentation="Time to get from remote backends synchronously(ms)",
            labelnames=labelnames,
            buckets=remote_time_to_get_sync,
        )

        # Ping latency metrics: use a gauge to record the latest ping latency
        self.gauge_remote_ping_latency = self._gauge_cls(
            name="lmcache:remote_ping_latency",
            documentation="Latest ping latency to remote backends (ms)",
            labelnames=labelnames,
        )
        self.counter_remote_ping_errors = self._counter_cls(
            name="lmcache:remote_ping_errors",
            documentation="Number of ping errors to remote backends",
            labelnames=labelnames,
        )
        self.counter_remote_ping_successes = self._counter_cls(
            name="lmcache:remote_ping_successes",
            documentation="Number of ping successes to remote backends",
            labelnames=labelnames,
        )
        self.gauge_remote_ping_error_code = self._gauge_cls(
            name="lmcache:remote_ping_error_code",
            documentation="Latest ping error code to remote backends",
            labelnames=labelnames,
        )

    def _log_gauge(self, gauge, data: Union[int, float]) -> None:
        # Convenience function for logging to gauge.
        gauge.set(data)

    def _log_counter(self, counter, data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        # Prevent ValueError from negative increment
        if data < 0:
            return
        counter.inc(data)

    def _log_histogram(self, histogram, data: Union[List[int], List[float]]) -> None:
        # Convenience function for logging to histogram.
        for value in data:
            histogram.observe(value)

    def convert_to_string(self, token_ids):
        return token_ids ## TODO
    
    def _log_hash_mapping(self, hash_token_mapping_batch):
        for key in hash_token_mapping_batch:
            self._hash_token_mapping[key] = self.convert_to_string(hash_token_mapping_batch[key])
    
    def _log_cache_events(self, cache_events):
        for (event, chunk_hash, key) in cache_events:
            self._cache_events.append((event, chunk_hash, key))
            if chunk_hash not in self._cache_event_counter_by_hash:
                self._cache_event_counter_by_hash[chunk_hash] = {
                    CacheEvent.STORE.name : 0,
                    CacheEvent.BLOCKING_HIT.name : 0,
                    CacheEvent.EVICT.name : 0,
                    CacheEvent.HIT.name : 0,
                    CacheEvent.NON_BLOCKING_HIT.name : 0,
                }
            self._cache_event_counter_by_hash[chunk_hash][event] += 1

    def log_json(self, stats: LMCacheStats):
        self._log_hash_mapping(stats.hash_chunk_mapping)
        self._log_cache_events(stats.cache_events)
        self._log_counter(
            self.counter_num_retrieve_requests, stats.interval_retrieve_requests
        )
        self._log_counter(
            self.counter_num_store_requests, stats.interval_store_requests
        )

        self._log_counter(
            self.counter_num_requested_tokens, stats.interval_requested_tokens
        )
        self._log_counter(self.counter_num_hit_tokens, stats.interval_hit_tokens)

        self._log_counter(
            self.counter_num_remote_read_requests,
            stats.interval_remote_read_requests,
        )
        self._log_counter(
            self.counter_num_remote_read_bytes, stats.interval_remote_read_bytes
        )
        self._log_counter(
            self.counter_num_remote_write_requests,
            stats.interval_remote_write_requests,
        )
        self._log_counter(
            self.counter_num_remote_write_bytes,
            stats.interval_remote_write_bytes,
        )

        self._log_gauge(self.gauge_cache_hit_rate, stats.cache_hit_rate)

        self._log_gauge(self.gauge_local_cache_usage, stats.local_cache_usage_bytes)

        self._log_gauge(self.gauge_remote_cache_usage, stats.remote_cache_usage_bytes)

        self._log_gauge(self.gauge_local_storage_usage, stats.local_storage_usage_bytes)

        self._log_histogram(self.histogram_time_to_retrieve, stats.time_to_retrieve)

        self._log_histogram(self.histogram_time_to_store, stats.time_to_store)

        self._log_histogram(self.histogram_retrieve_speed, stats.retrieve_speed)

        self._log_histogram(self.histogram_store_speed, stats.store_speed)

        self._log_histogram(
            self.histogram_remote_time_to_get, stats.interval_remote_time_to_get
        )
        self._log_histogram(
            self.histogram_remote_time_to_put, stats.interval_remote_time_to_put
        )
        self._log_histogram(
            self.histogram_remote_time_to_get_sync,
            stats.interval_remote_time_to_get_sync,
        )
        self._log_gauge(
            self.gauge_remote_ping_latency, stats.interval_remote_ping_latency
        )
        self._log_counter(
            self.counter_remote_ping_errors, stats.interval_remote_ping_errors
        )
        self._log_counter(
            self.counter_remote_ping_successes, stats.interval_remote_ping_success
        )
        self._log_gauge(
            self.gauge_remote_ping_error_code, stats.interval_remote_ping_error_code
        )

        self._commit_json()

    @staticmethod
    def _metadata_to_labels(metadata: LMCacheEngineMetadata):
        return {
            "model_name": metadata.model_name,
            "worker_id": metadata.worker_id,
        }

    _instance = None

    @staticmethod
    def GetOrCreate(metadata: LMCacheEngineMetadata) -> "PrometheusLogger":
        if JsonLogger._instance is None:
            JsonLogger._instance = JsonLogger(metadata)

        if JsonLogger._instance.metadata != metadata:
            logger.error(
                "JsonLogger instance already created with"
                "different metadata. This should not happen except "
                "in test"
            )
        return JsonLogger._instance

    @staticmethod
    def GetInstance() -> "JsonLogger":
        assert JsonLogger._instance is not None, (
            "PrometheusLogger instance not created yet"
        )
        return JsonLogger._instance

    @staticmethod
    def GetInstanceOrNone() -> Optional["PrometheusLogger"]:
        """
        Returns the singleton instance of PrometheusLogger if it exists,
        otherwise returns None.
        """
        return PrometheusLogger._instance


class LMCacheStatsLogger:
    def __init__(self, metadata: LMCacheEngineMetadata, log_interval: int):
        self.metadata = metadata
        self.log_interval = log_interval
        self.monitor = LMCStatsMonitor.GetOrCreate()
        self.prometheus_logger = JsonLogger.GetOrCreate(metadata)
        self.is_running = True

        self.thread = threading.Thread(target=self.log_worker, daemon=True)
        self.thread.start()

    def log_worker(self):
        while self.is_running:
            stats = self.monitor.get_stats_and_clear()
            if "LMCACHE_ENABLE_CACHE_LOGGING" in os.environ and (
                int(os.environ["LMCACHE_ENABLE_CACHE_LOGGING"]) == 1
                 or int(os.environ["LMCACHE_ENABLE_CACHE_LOGGING"]) == 2):
                self.prometheus_logger.log_json(stats)
            time.sleep(self.log_interval)

    def shutdown(self):
        self.is_running = False
        self.thread.join()

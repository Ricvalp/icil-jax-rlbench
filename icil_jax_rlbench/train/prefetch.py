from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class _WorkerError:
    exc: BaseException
    traceback_text: str


class BackgroundBatchPrefetcher:
    """Build batches in background threads and expose a blocking get()."""

    def __init__(
        self,
        make_batch_fn: Callable[[int], Callable[[], Any]],
        *,
        num_workers: int,
        max_prefetch: int,
    ) -> None:
        self.num_workers = max(1, int(num_workers))
        self.max_prefetch = max(1, int(max_prefetch))
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.max_prefetch)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        for worker_id in range(self.num_workers):
            batch_fn = make_batch_fn(worker_id)
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id, batch_fn),
                name=f'icil-batch-prefetch-{worker_id}',
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _put(self, item: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def _worker_loop(self, worker_id: int, batch_fn: Callable[[], Any]) -> None:
        del worker_id
        while not self._stop.is_set():
            try:
                batch = batch_fn()
            except BaseException as exc:
                self._put(_WorkerError(exc=exc, traceback_text=traceback.format_exc()))
                return
            self._put(batch)

    def get(self) -> Any:
        item = self._queue.get()
        if isinstance(item, _WorkerError):
            self.close()
            raise RuntimeError(f'Batch prefetch worker failed:\n{item.traceback_text}') from item.exc
        return item

    def qsize(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.0)


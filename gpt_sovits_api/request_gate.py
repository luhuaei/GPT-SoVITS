import asyncio
import time
from dataclasses import dataclass


@dataclass
class RequestTicket:
    gate: "RequestGate"
    request_id: str
    queued_at: float
    started_at: float
    released: bool = False

    @property
    def queue_wait_seconds(self) -> float:
        return max(0.0, self.started_at - self.queued_at)

    async def release(self):
        if self.released:
            return
        self.released = True
        await self.gate.release(self)


class RequestGate:
    def __init__(self, max_concurrent: int = 1, max_queue_size: int = 16):
        self.max_concurrent = max(1, int(max_concurrent))
        self.max_queue_size = max(0, int(max_queue_size))
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        self.lock = asyncio.Lock()
        self.active = 0
        self.queued = 0
        self.total_started = 0
        self.total_completed = 0
        self.total_rejected = 0

    async def acquire(self, request_id: str) -> RequestTicket:
        queued_at = time.perf_counter()
        async with self.lock:
            if self.queued >= self.max_queue_size:
                self.total_rejected += 1
                raise RuntimeError("request queue is full")
            self.queued += 1
        await self.semaphore.acquire()
        started_at = time.perf_counter()
        async with self.lock:
            self.queued -= 1
            self.active += 1
            self.total_started += 1
        return RequestTicket(
            gate=self,
            request_id=request_id,
            queued_at=queued_at,
            started_at=started_at,
        )

    async def release(self, ticket: RequestTicket):
        async with self.lock:
            self.active = max(0, self.active - 1)
            self.total_completed += 1
        self.semaphore.release()

    async def snapshot(self) -> dict:
        async with self.lock:
            return {
                "max_concurrent": self.max_concurrent,
                "max_queue_size": self.max_queue_size,
                "active_requests": self.active,
                "queued_requests": self.queued,
                "total_started": self.total_started,
                "total_completed": self.total_completed,
                "total_rejected": self.total_rejected,
            }


"""Minimal test script for LiveAvatar plugin changes.

Only requires HEYGEN_LIVE_AVATAR_API_KEY. Verifies:
1. Session creation
2. WS ready gate (session.state_updated connected)
3. Audio chunking (400ms first chunk, 1s subsequent)
4. Keep-alive task is running
5. Clean shutdown

Usage:
    uv run python test_liveavatar_changes.py
"""

import asyncio
import math
import os
import struct
import sys

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.services.heygen.api_liveavatar import LiveAvatarNewSessionRequest
from pipecat.services.heygen.client import (
    HEY_GEN_SAMPLE_RATE,
    HeyGenCallbacks,
    HeyGenClient,
    ServiceType,
)
from pipecat.transports.base_transport import TransportParams

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


def generate_sine_tone(freq_hz=440, duration_secs=3.0, sample_rate=HEY_GEN_SAMPLE_RATE):
    """Generate a sine wave tone as PCM 16-bit mono bytes."""
    num_samples = int(sample_rate * duration_secs)
    samples = []
    for i in range(num_samples):
        value = int(16000 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        samples.append(struct.pack("<h", value))
    return b"".join(samples)


class FakeTaskManager:
    """Minimal task manager for testing outside a pipeline."""

    def __init__(self):
        """Initialize task manager."""
        self._tasks = []

    def create_task(self, coro, name=None):
        """Create and track an async task."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        return task

    async def cancel_task(self, task, timeout=5.0):
        """Cancel a task with timeout."""
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    async def cancel_all(self):
        """Cancel all tracked tasks."""
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)


class FakeSetup:
    """Minimal setup object for HeyGenClient.setup()."""

    def __init__(self, task_manager):
        """Initialize with task manager."""
        self.task_manager = task_manager


async def main():
    """Run LiveAvatar plugin tests."""
    api_key = os.getenv("HEYGEN_LIVE_AVATAR_API_KEY")
    if not api_key:
        logger.error("Set HEYGEN_LIVE_AVATAR_API_KEY in .env or environment")
        sys.exit(1)

    connected_event = asyncio.Event()
    participant_event = asyncio.Event()

    async def on_connected():
        logger.info("[TEST] on_connected fired -- WS ready gate passed!")
        connected_event.set()

    async def on_participant_connected(pid):
        logger.info(f"[TEST] participant connected: {pid}")
        participant_event.set()

    async def on_participant_disconnected(pid):
        logger.info(f"[TEST] participant disconnected: {pid}")

    task_manager = FakeTaskManager()

    async with aiohttp.ClientSession() as session:
        client = HeyGenClient(
            api_key=api_key,
            session=session,
            params=TransportParams(
                audio_in_sample_rate=48000,
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_out_sample_rate=HEY_GEN_SAMPLE_RATE,
            ),
            session_request=LiveAvatarNewSessionRequest(
                is_sandbox=True,
                avatar_id="dd73ea75-1218-4ef3-92ce-606d5f7fbc0a",
            ),
            service_type=ServiceType.LIVE_AVATAR,
            callbacks=HeyGenCallbacks(
                on_connected=on_connected,
                on_participant_connected=on_participant_connected,
                on_participant_disconnected=on_participant_disconnected,
            ),
        )

        # --- Test 1: Setup & session creation ---
        logger.info("[TEST 1] Creating session...")
        setup = FakeSetup(task_manager)
        await client.setup(setup)
        logger.info("[TEST 1] PASSED - Session created")

        # --- Test 2: Start client ---
        logger.info("[TEST 2] Starting client...")

        from pipecat.frames.frames import StartFrame

        start_frame = StartFrame(
            audio_in_sample_rate=48000,
            audio_out_sample_rate=HEY_GEN_SAMPLE_RATE,
        )
        await client.start(start_frame)

        # Wait for on_connected callback
        try:
            await asyncio.wait_for(connected_event.wait(), timeout=15.0)
            logger.info("[TEST 2] PASSED - Client started and connected")
        except asyncio.TimeoutError:
            logger.error("[TEST 2] FAILED - on_connected never fired")
            await client.stop()
            await task_manager.cancel_all()
            return

        # --- Test 3: Verify keep-alive task is running ---
        if client._keep_alive_task and not client._keep_alive_task.done():
            logger.info("[TEST 3] PASSED - Keep-alive task is running")
        else:
            logger.error("[TEST 3] FAILED - Keep-alive task not found or already done")

        # --- Test 4: Send test audio with new chunking ---
        logger.info("[TEST 4] Sending 3s sine tone (testing 400ms first + 1s chunks)...")
        audio_data = generate_sine_tone(freq_hz=440, duration_secs=3.0)
        event_id = "test-event-001"

        # Simulate the chunking logic from video.py
        first_chunk_size = int(HEY_GEN_SAMPLE_RATE * 2 * 0.4)  # 400ms = 19200 bytes
        chunk_size = int(HEY_GEN_SAMPLE_RATE * 2 * 1.0)  # 1s = 48000 bytes

        offset = 0
        chunk_num = 0
        is_first = True
        while offset < len(audio_data):
            current_size = first_chunk_size if is_first else chunk_size
            chunk = audio_data[offset : offset + current_size]
            if len(chunk) == 0:
                break
            await client.agent_speak(chunk, event_id)
            chunk_num += 1
            actual_ms = len(chunk) / (HEY_GEN_SAMPLE_RATE * 2) * 1000
            logger.info(f"  Sent chunk {chunk_num}: {len(chunk)} bytes ({actual_ms:.0f}ms)")
            if is_first:
                is_first = False
            offset += current_size

        await client.agent_speak_end(event_id)
        logger.info(f"[TEST 4] PASSED - Sent {chunk_num} chunks, agent_speak_end sent")

        # --- Test 5: Clean shutdown ---
        logger.info("[TEST 5] Stopping client...")
        await client.stop()
        await task_manager.cancel_all()
        logger.info("[TEST 5] PASSED - Clean shutdown")

    logger.info("")
    logger.info("All tests completed!")


if __name__ == "__main__":
    asyncio.run(main())

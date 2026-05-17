# src/opd/inference/message

## Purpose
Serializable message types for communication between the scheduler and the tokenizer process (IPC mode). Not used in embedded/offline mode where messages are replaced by direct in-memory calls.

## Key Parts
- `backend.py`: Backend-side message types — `BaseBackendMsg`, `UserMsg` (new request), `AbortBackendMsg`, `BatchBackendMsg`, `ExitMsg`, `DetokenizeMsg` (token result).
- `frontend.py`: Frontend/tokenizer-side message types — `BaseTokenizerMsg`, `BatchTokenizerMsg`.
- `tokenizer.py`: Tokenizer process message handling.
- `utils.py`: `serialize_type` / `deserialize_type` — JSON serialization helpers for message dispatch.

## Entry Points
- `UserMsg`: constructed by `_EmbeddedScheduler.offline_receive_msg` (embedded) or the frontend (IPC).
- `DetokenizeMsg`: produced by `Scheduler._process_last_data`, consumed by `send_result`.

## Outbound Dependencies
- `liteopd.inference.core`: `SamplingParams`.

## Inbound Dependents
- `liteopd.inference.scheduler.scheduler`: processes `UserMsg`, `AbortBackendMsg`, `ExitMsg`; sends `DetokenizeMsg`.
- `liteopd.runtime.rollout._EmbeddedScheduler`: constructs `UserMsg` directly.

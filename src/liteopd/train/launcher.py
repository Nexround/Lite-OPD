from __future__ import annotations

import argparse
from datetime import datetime
import os
import shutil
import signal
import subprocess
import time
import traceback
from pathlib import Path

from liteopd.train.config import load_train_config
from liteopd.train.logging import JsonlLogger


class _LauncherSignalGuard:
    def __init__(self, on_signal=None) -> None:
        self._on_signal = on_signal
        self._previous_handlers: dict[int, object] = {}

    def __enter__(self):
        self._previous_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }

        def _handler(sig, frame):
            if self._on_signal is not None:
                self._on_signal(sig)
            if sig == signal.SIGINT:
                raise KeyboardInterrupt
            raise SystemExit(128 + sig)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
        return self

    def __exit__(self, exc_type, exc, tb):
        signal.signal(signal.SIGINT, self._previous_handlers[signal.SIGINT])
        signal.signal(signal.SIGTERM, self._previous_handlers[signal.SIGTERM])
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    args = parser.parse_args()

    os.environ["WORLD_SIZE"] = str(args.nproc_per_node)
    cfg = load_train_config(args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.output_dir = str(Path(cfg.output_dir) / timestamp)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = out_dir / Path(args.config).name
    shutil.copy2(args.config, config_snapshot_path)
    wrapper_logger = JsonlLogger(out_dir / "launcher_log.jsonl")
    wrapper_logger.log({
        "phase": "launcher_start",
        "time": time.time(),
        "config_path": args.config,
        "output_dir": cfg.output_dir,
        "config_snapshot_path": str(config_snapshot_path),
        "nproc_per_node": args.nproc_per_node,
    })

    train_process = None
    cleanup_started = False
    env = os.environ.copy()
    env["OPD_OUTPUT_DIR"] = cfg.output_dir

    import sys
    train_command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        "-m",
        "liteopd.train.run_opd_training",
        "--config",
        args.config,
    ]

    def _cleanup_launcher_runtime() -> None:
        nonlocal cleanup_started, train_process
        if cleanup_started:
            return
        cleanup_started = True
        wrapper_logger.log({
            "phase": "launcher_cleanup_start",
            "time": time.time(),
            "has_train_process": train_process is not None,
        })
        if train_process is not None and train_process.poll() is None:
            try:
                train_process.terminate()
                train_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                train_process.kill()
                train_process.wait(timeout=10)
            except Exception:
                wrapper_logger.log({
                    "phase": "launcher_train_process_terminate_failed",
                    "time": time.time(),
                    "traceback": traceback.format_exc(),
                })
        wrapper_logger.log({"phase": "launcher_cleanup_done", "time": time.time()})

    def _on_signal(sig: int) -> None:
        wrapper_logger.log({
            "phase": "launcher_signal_received",
            "time": time.time(),
            "signal": sig,
        })
        _cleanup_launcher_runtime()

    try:
        with _LauncherSignalGuard(on_signal=_on_signal):
            wrapper_logger.log({
                "phase": "launcher_torchrun_start",
                "time": time.time(),
                "command": train_command,
            })
            train_process = subprocess.Popen(train_command, env=env)
            returncode = train_process.wait()
            if returncode != 0:
                raise subprocess.CalledProcessError(returncode=returncode, cmd=train_command)
            wrapper_logger.log({
                "phase": "launcher_torchrun_success",
                "time": time.time(),
                "returncode": returncode,
            })
    except subprocess.CalledProcessError as exc:
        wrapper_logger.log({
            "phase": "launcher_torchrun_failed",
            "time": time.time(),
            "returncode": exc.returncode,
            "command": exc.cmd,
            "traceback": traceback.format_exc(),
        })
        raise
    except Exception:
        wrapper_logger.log({
            "phase": "launcher_exception",
            "time": time.time(),
            "traceback": traceback.format_exc(),
        })
        raise
    finally:
        _cleanup_launcher_runtime()


if __name__ == "__main__":
    main()

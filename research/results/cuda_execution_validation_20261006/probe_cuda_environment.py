#!/usr/bin/env python3
"""Allowlisted environment metadata; optional tiny CUDA smoke, never simulation."""
import argparse
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sys
import warnings

sys.dont_write_bytecode = True

SCHEMA = "adaptgns_cuda_environment_probe_v1"
# Imports are fixed here, never taken from user input. No repository module imports.
DEPENDENCIES = (
    ("numpy", "numpy", "core", ()),
    ("scipy", "scipy", "core", ()),
    ("scipy.spatial", "scipy", "core", ("cKDTree",)),
    ("torch", "torch", "core", ()),
    ("torch_geometric", "torch-geometric", "core", ()),
    ("torch_geometric.nn", "torch-geometric", "core", ("MessagePassing", "radius_graph")),
    ("torch_cluster", "torch-cluster", "optional_pyg_radius_backend", ()),
    ("torch_scatter", "torch-scatter", "optional_pyg_extension", ()),
    ("torch_sparse", "torch-sparse", "optional_pyg_extension", ()),
    ("absl.flags", "absl-py", "legacy_cli", ()),
    ("tqdm", "tqdm", "legacy_cli", ()),
    ("tfrecord", "tfrecord", "data_conversion", ()),
    ("google.protobuf", "protobuf", "data_conversion", ()),
    ("crc32c", "crc32c", "data_conversion", ()),
)


def error_kind(error):
    # Exception text can contain local paths or other incidental system details.
    return type(error).__name__


def import_dependencies():
    records, modules = [], {}
    for name, distribution, role, symbols in DEPENDENCIES:
        record = {"module": name, "role": role, "status": "unavailable"}
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    module = importlib.import_module(name)
                    for symbol in symbols:
                        getattr(module, symbol)
                    modules[name] = module
                    record["status"] = "imported"
                    try:
                        record["version"] = importlib.metadata.version(distribution)
                    except importlib.metadata.PackageNotFoundError:
                        record["version"] = None
                    record["symbols_importable"] = list(symbols)
                except Exception as error:
                    record["error_type"] = error_kind(error)
                record["python_import_diagnostics_suppressed"] = bool(caught or stdout.getvalue() or stderr.getvalue())
        records.append(record)
    return records, modules


def sysconf_bytes(pages_name):
    try:
        pages, size = os.sysconf(pages_name), os.sysconf("SC_PAGE_SIZE")
        return int(pages * size) if pages > 0 and size > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def host_metadata():
    host = {
        "os": platform.system(), "os_release": platform.release(),
        "architecture": platform.machine(), "cpu_description": platform.processor() or None,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(), "logical_cpu_count": os.cpu_count(),
        "ram_physical_bytes": sysconf_bytes("SC_PHYS_PAGES"),
        "ram_available_pages_bytes": sysconf_bytes("SC_AVPHYS_PAGES"),
        "ram_scope": "OS page counters; not a cgroup or scheduler allocation guarantee",
    }
    try:
        host["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        host["cpu_affinity_count"] = None
    try:
        disk = shutil.disk_usage(".")
        host["disk"] = {"scope": "current working directory filesystem", "total_bytes": disk.total,
                        "used_bytes": disk.used, "free_bytes": disk.free}
    except OSError as error:
        host["disk"] = {"error_type": error_kind(error)}
    return host


def cuda_metadata(torch):
    if torch is None:
        return {"status": "torch_unavailable", "available": False, "devices": []}
    cuda = {"status": "queried", "devices": [],
            "version_scope": "PyTorch build CUDA version; does not query installed toolkit or driver version"}
    try:
        cuda["pytorch_version"] = str(torch.__version__)
        cuda["pytorch_cuda_build_version"] = torch.version.cuda
        cuda["built_with_cuda"] = bool(torch.backends.cuda.is_built())
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["visible_device_count"] = int(torch.cuda.device_count())
        cuda["compiled_architectures"] = list(torch.cuda.get_arch_list())
        if cuda["available"]:
            for index in range(cuda["visible_device_count"]):
                properties = torch.cuda.get_device_properties(index)
                cuda["devices"].append({"visible_index": index, "name": properties.name,
                                        "compute_capability": [properties.major, properties.minor],
                                        "total_memory_bytes": int(properties.total_memory),
                                        "multiprocessor_count": int(properties.multi_processor_count)})
    except Exception as error:
        cuda.update(status="query_failed", available=False, error_type=error_kind(error))
    return cuda


def synthetic_cuda_smoke(torch, cuda):
    """Use only visible CUDA device0; fork and explicitly restore CPU/CUDA RNG."""
    if torch is None or not cuda.get("available"):
        return {"status": "not_run", "reason": "CUDA unavailable; no CPU or MPS fallback"}
    result = {"status": "failed", "visible_device_index": 0, "dtype": "float32",
              "matrix_shape": [32, 32], "optimizer_updates": 0}
    cpu_before = cuda_before = None
    try:
        cpu_before = torch.get_rng_state().clone()
        cuda_before = torch.cuda.get_rng_state(0).clone()
        with torch.random.fork_rng(devices=[0], enabled=True), torch.cuda.device(0):
            torch.cuda.manual_seed(20261006)
            state = torch.cuda.get_rng_state(0).clone()
            first = torch.rand((16, 16), device="cuda:0")
            torch.cuda.set_rng_state(state, 0)
            second = torch.rand((16, 16), device="cuda:0")
            result["cuda_rng_replay_equal"] = bool(torch.equal(first, second))
            x = (torch.arange(1024, device="cuda:0", dtype=torch.float32).reshape(32, 32) / 1024).requires_grad_()
            weight = torch.linspace(-0.1, 0.1, 1024, device="cuda:0", dtype=torch.float32).reshape(32, 32).requires_grad_()
            prediction = x @ weight
            loss = prediction.square().mean()
            loss.backward()
            torch.cuda.synchronize(0)
            result["finite_output_and_loss"] = bool(torch.isfinite(prediction).all() and torch.isfinite(loss))
            result["finite_nonzero_gradients"] = all(
                tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
                and bool(torch.count_nonzero(tensor.grad)) for tensor in (x, weight))
            result["synthetic_loss"] = float(loss.detach().cpu()) if bool(torch.isfinite(loss)) else None
    except Exception as error:
        result["error_type"] = error_kind(error)
    finally:
        try:
            result["cpu_rng_restored"] = cpu_before is not None and bool(torch.equal(cpu_before, torch.get_rng_state()))
            result["cuda_rng_restored"] = cuda_before is not None and bool(torch.equal(cuda_before, torch.cuda.get_rng_state(0)))
        except Exception as error:
            result["rng_verification_error_type"] = error_kind(error)
    checks = ("cuda_rng_replay_equal", "finite_output_and_loss", "finite_nonzero_gradients",
              "cpu_rng_restored", "cuda_rng_restored")
    if all(result.get(name) is True for name in checks) and "error_type" not in result:
        result["status"] = "passed"
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--metadata-only", action="store_true", help="Default: imports and allowlisted metadata only; no tensor smoke")
    mode.add_argument("--synthetic-smoke", action="store_true", help="Also run tiny synthetic CUDA tensors, gradients and RNG replay on visible device0")
    return parser.parse_args(argv)


def collect_report(run_smoke=False):
    records, modules = import_dependencies()
    # Suppress Python-level diagnostics; native extensions can write directly to FDs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cuda = cuda_metadata(modules.get("torch"))
        smoke = synthetic_cuda_smoke(modules.get("torch"), cuda) if run_smoke else {"status": "not_requested"}
    return {"schema": SCHEMA, "mode": "synthetic-smoke" if run_smoke else "metadata-only",
            "host": host_metadata(), "dependencies": records, "cuda": cuda, "synthetic_smoke": smoke,
            "scope": "Environment triage only; not simulator, graph-kernel, checkpoint, resume, performance or scientific validation",
            "missing_core_imports": [row["module"] for row in records if row["role"] == "core" and row["status"] != "imported"]}


def exit_code(report):
    # Metadata inventory succeeds even when dependencies/CUDA are absent.
    if report["mode"] == "metadata-only":
        return 0
    return {"passed": 0, "not_run": 2}.get(report["synthetic_smoke"]["status"], 1)


def main(argv=None):
    args = parse_args(argv)
    report = collect_report(run_smoke=args.synthetic_smoke)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())

import glob
import os


def _resolve_and_validate(path, source):
    resolved = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"ONNX path from {source} does not exist: {resolved}")
    return resolved


def resolve_onnx_path(project_root, cli_onnx, env_vars, onnx_glob, robot_name):
    """Resolve ONNX path with priority: CLI > env var > latest matched glob."""
    if cli_onnx:
        return _resolve_and_validate(cli_onnx, "--onnx")

    for env_name in env_vars:
        env_value = os.environ.get(env_name)
        if env_value:
            return _resolve_and_validate(env_value, f"${env_name}")

    pattern = os.path.join(project_root, "onnx", onnx_glob)
    candidates = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    if candidates:
        # Newest by mtime; tie-break by basename for deterministic behavior.
        candidates.sort(key=lambda p: (os.path.getmtime(p), os.path.basename(p)))
        return candidates[-1]

    raise FileNotFoundError(
        f"No ONNX found for {robot_name}. "
        f"Checked env vars {env_vars} and glob {pattern}. "
        f"Please pass --onnx /abs/path/model.onnx."
    )


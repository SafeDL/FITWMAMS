"""Keep experiment paths readable while storing model bytes on the model volume."""

import shutil
from pathlib import Path

MODEL_ROOT = Path("/media/hp/DATA/nuplan_workspace/FITWMAMS_models")


def model_destination(output, logical, *, root=None):
    run = output.resolve()
    namespace = (
        Path("history", run.name) if run.parent.name == "history" else Path(run.name)
    )
    relative = logical.relative_to(output)
    if relative.parts[0] == "manual_nuplan":
        relative = Path("native", *relative.parts[1:])
    else:
        method = relative.parts[1]
        relative = (
            Path("checkpoints")
            / method
            / relative.relative_to(Path("repairs") / method / "candidates")
        )
    destination = (
        (Path(root) if root is not None else MODEL_ROOT)
        / "ads"
        / "diffusion_planner"
        / namespace
        / relative
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def link_checkpoint(logical, actual):
    logical.parent.mkdir(parents=True, exist_ok=True)
    temporary = logical.with_name(logical.name + ".link.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(actual.resolve())
    temporary.replace(logical)
    if (actual.parent / "args.json").exists():
        shutil.copy2(actual.parent / "args.json", logical.parent / "args.json")


def save_checkpoint(policy, output, logical, *, root=None):
    actual = model_destination(output, logical, root=root)
    temporary = actual.with_name(actual.name + ".tmp")
    policy.save(temporary)
    temporary.replace(actual)
    link_checkpoint(logical, actual)

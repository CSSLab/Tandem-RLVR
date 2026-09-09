"""Report the installed stack: versions, import locations, visible GPUs.

Run after env/install.sh and the fork installs in docs/INSTALL.md. The import
locations matter as much as the versions: if vllm or verl resolves to
site-packages rather than the patched clone, the fork is not what runs, and the
tandem rollout degrades to stock behaviour without raising anything.
"""

import importlib
import sys

PACKAGES = ("torch", "vllm", "verl")


def main() -> int:
    loaded = {}
    failed = []
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            failed.append(name)
            print(f"{name:12s} IMPORT FAILED  {type(exc).__name__}: {exc}")
            continue
        loaded[name] = module
        version = getattr(module, "__version__", "no __version__")
        print(f"{name:12s} {version}")
        print(f"{'':12s} from {getattr(module, '__file__', 'unknown location')}")

    torch = loaded.get("torch")
    if torch is None:
        print("gpus         unknown, torch did not import")
    elif not torch.cuda.is_available():
        print("gpus         0, torch.cuda.is_available() is False")
    else:
        count = torch.cuda.device_count()
        print(f"gpus         {count} (torch cuda build {torch.version.cuda})")
        for i in range(count):
            print(f"{'':12s} cuda:{i} {torch.cuda.get_device_name(i)}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}. See env/SETUP.md.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

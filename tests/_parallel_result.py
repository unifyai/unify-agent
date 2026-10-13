"""Publish one run-owned pytest exit receipt before its tmux session closes."""

import os
import sys
import tempfile
from pathlib import Path


def publish(path: Path, status: int) -> None:
    if type(status) is not int or not 0 <= status <= 255:
        raise ValueError("invalid process exit status")
    if not path.is_absolute() or path.parent.is_symlink():
        raise ValueError("receipt directory must be an absolute owned directory")
    descriptor, temporary = tempfile.mkstemp(prefix=".exit-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"{status}\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Create-only publication: neither an existing receipt nor a symlink
        # can be replaced. The controller retains the target/occurrence map.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


if __name__ == "__main__":
    publish(Path(sys.argv[1]), int(sys.argv[2]))

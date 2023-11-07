from __future__ import annotations
import pathlib


def print_files(files: dict[str, pathlib.Path]):
    """Print files for snakemake rule."""
    print(',\n'.join(f'{k}="{v.as_posix()}"' for k, v in files.items()))

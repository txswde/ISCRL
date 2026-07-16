"""Inspect the schema and shapes of an ISCRL HDF5 feature file."""

from __future__ import annotations

import argparse

import h5py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to an HDF5 feature file")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    with h5py.File(args.dataset, "r") as dataset:
        keys = list(dataset.keys())
        print(f"Videos: {len(keys)}")
        for key in keys[: args.limit]:
            print(f"-- {key} --")
            group = dataset[key]
            for name, value in group.items():
                print(f"  {name}: shape={value.shape}, dtype={value.dtype}")
            if group.attrs:
                print(f"  attributes: {dict(group.attrs)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compatibility entrypoint for the audited, final-roster readout figure.

Historical figure labels are resolved by the final submission assembly.
All plotting logic lives in figure_immune_readouts.py.
"""
from figure_immune_readouts import main as build


def main():
    build(["--figures", "S3"])


if __name__ == "__main__":
    main()

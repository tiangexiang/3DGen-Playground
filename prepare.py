"""Prepare text tokens and camera framing using the default training configuration."""
import sys
from mini3d.cli import main

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])

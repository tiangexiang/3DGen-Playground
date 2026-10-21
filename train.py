"""Train the default Gaussian Splat Diffusion baseline; use --config-file or --set for overrides."""
import sys
from mini3d.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])

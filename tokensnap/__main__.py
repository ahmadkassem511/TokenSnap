"""Allow `python -m tokensnap` to behave like the `tokensnap` console script."""

from tokensnap.cli import app

if __name__ == "__main__":
    app()

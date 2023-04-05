import argparse
from src.utils._xfer_tools import cleanup


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', default='/scratch.local')
    args_parsed = parser.parse_args()

    cleanup(args_parsed.destination)

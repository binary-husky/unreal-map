import os, sys
import argparse
from VISUALIZE.mcom import *
from VISUALIZE.mcom_replay import RecallProcessThreejs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-p', '--path', help='directory of chosen file')
    args, unknown = parser.parse_known_args()
    if hasattr(args, 'path'):
        path = args.path
    else:
        assert False, (r"parser.add_argument('-p', '--path', help='The node name is?')")

    load_via_json = (hasattr(args, 'cfg') and args.cfg is not None)
    
    rp = RecallProcessThreejs(path)
    rp.start()
    rp.join()

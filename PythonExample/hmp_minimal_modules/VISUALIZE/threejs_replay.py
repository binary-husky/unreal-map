import os, sys
import argparse
from VISUALIZE.mcom import *
from VISUALIZE.mcom_replay import RecallProcessThreejs
from UTIL.network import find_free_port


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HMP')
    parser.add_argument('-f', '--file', help='Directory of chosen file', default='TEMP/v2d_logger/backup.dp.gz')
    parser.add_argument('-p', '--port', help='The port for web server')
    args, unknown = parser.parse_known_args()
    if hasattr(args, 'file'):
        path = args.file
    else:
        assert False, (r"parser.add_argument('-f', '--file', help='The node name is?')")

    if hasattr(args, 'port') and args.port is not None:
        port = int(args.port)
    else:
        port = find_free_port()
        print('no --port arg, auto find:', port)

    load_via_json = (hasattr(args, 'cfg') and args.cfg is not None)
    
    rp = RecallProcessThreejs(path, port)
    rp.start()
    rp.join()


'''

note=RVE-drone1-fixaa-run2
cp -r ./ZHECKPOINT/$note ./ZHECKPOINT/$note-bk
cp -r ./ZHECKPOINT/$note/experiment.jsonc ./ZHECKPOINT/$note/experiment-bk.jsonc
cp -r ./ZHECKPOINT/$note/experiment.jsonc ./ZHECKPOINT/$note/train.jsonc
cp -r ./ZHECKPOINT/$note/experiment.jsonc ./ZHECKPOINT/$note/test.jsonc

python << __EOF__
import commentjson as json
file = "./ZHECKPOINT/$note/test.jsonc"
print(file)
with open(file, encoding='utf8') as f:
    json_data = json.load(f)
json_data["config.py->GlobalConfig"]["num_threads"] = 1
json_data["config.py->GlobalConfig"]["fold"] = 1
json_data["config.py->GlobalConfig"]["test_only"] = True
json_data["MISSION.uhmap.uhmap_env_wrapper.py->ScenarioConfig"]["TimeDilation"] = 1
json_data["ALGORITHM.conc_4hist_hete.foundation.py->AlgorithmConfig"]["load_checkpoint"] = True
with open(file, 'w') as f:
    json.dump(json_data, f, indent=4)
__EOF__

python main.py -c ./ZHECKPOINT/$note/test.jsonc




'''
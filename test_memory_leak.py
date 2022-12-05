import os, sys
os.chdir('./Build')
if not os.path.exists('./hmp2g'):
    os.system('git clone https://github.com/binary-husky/hmp2g.git -b uhmap-memleak-test')
os.chdir('./hmp2g')
os.system('python main.py -c ZHECKPOINT/uhmap_hete10vs10/render_result.jsonc')

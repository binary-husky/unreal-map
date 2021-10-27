Author: Fu Qingxu,CASIA
running

## <1-1> all default: testing
git pull && python main.py -c ZHECKPOINT/50vs50-eval/test.json
git pull && python main.py -c ZHECKPOINT/100vs100-eval/test.json  # old alg version

## <1-2> all default: training 
git pull && python main.py -c train.json



## <2> change settings

launch with: 
python main.py --cfg xx.json


python main.py -c ZHECKPOINT/50vs50-eval/test.json
git pull && python main.py -c ZHECKPOINT/10

## <3> project road map
If you are interested in something, you may continue to read:

    Handling parallel environment             -->   task_runner.py & shm_env.py

    Link between teams and diverse algorithms -->   multi_team.py

    Adding new env                            -->   MISSIONS.env_router.py

    Adding algorithm                          -->   ALGORITHM.example_foundation.py

    Configuring by writing py files           -->   config.py

    Configuring by json                       -->   xx.json

    colorful printing                         -->   colorful.py

    auto pip deployer                         -->   pip_find_missing.py

    efficient parallel execting               -->   shm_pool.pyx

    auto gpu selection                        -->   auto_gpu.py

    matlab logging/plotting bridge            -->   mcom.py & mcom_rec.py

    experiment batch executor                 -->   mprofile.py

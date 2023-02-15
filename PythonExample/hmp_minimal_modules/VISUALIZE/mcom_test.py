def validate_path():
    import os, sys
    dir_name = os.path.dirname(__file__)
    root_dir_assume = os.path.abspath(os.path.dirname(__file__) +  '/..')
    os.chdir(root_dir_assume)
    sys.path.append(root_dir_assume)
    
validate_path() # validate path so you can run from base directory

from VISUALIZE.mcom import mcom

mcv = mcom(
    path='./TEMP',  # path to generate log
    draw_mode='Img',    # draw mode
    resume_mod=True,    # resume from previous session
    # figsize=(48,12),  # manual fig size
    resume_file='ZHECKPOINT/RVE-drone2-ppoma-run1/logger/mcom_buffer_0____starting_session.txt',   # pick up from a specific session txt
    image_path='./temp2.jpg',   # target image directory
    smooth_level=40,  # smooth line level
    # rec_exclude=["r*", "n*", 
    #     "*0.00*", 
    #     "*0.01*", 
    #     "*0.04*", 
    #     "*0.06*", 
    #     "*0.11*", 
    #     "*0.18*", 
    #     "*0.25*", 
    # ],
)

input('wait complete')


from VISUALIZE.mcom import mcom

mcv = mcom(
    path='./TEMP',
    draw_mode='Img',
    resume_mod=True,
    # figsize=(72,12),
    # resume_file='ZHECKPOINT/RVE-drone2-ppoma-run1/logger/mcom_buffer_0____starting_session.txt',
    resume_file='mcom_buffer_0____starting_session.txt',
    image_path='./temp.jpg',
    rec_exclude=["r*", "n*", "*0.16*", "*0.50*", 
    "*0.28*", "*0.08*", "*0.18*", "*0.41*", "*0.20*", "*0.07*", "*0.06*"
    ],
)
input('wait complete')
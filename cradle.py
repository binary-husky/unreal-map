from VISUALIZE.mcom import mcom

mcv = mcom(
    path='./TEMP',
    draw_mode='Img',
    resume_mod=True,
    # figsize=(72,12),
    # resume_file='ZHECKPOINT/RVE-drone2-ppoma-run1/logger/mcom_buffer_0____starting_session.txt',
    # resume_file='mcom_buffer_0____starting_session.txt',
    resume_file='x.txt',
    image_path='./temp.jpg',
    rec_exclude=["r*", "n*", "*0.16*", "*0.50*", 
    "*0.28*", "*0.08*", "*0.18*", "*0.41*", "*0.20*", "*0.07*", "*0.06*"
    ],
)

#####################################################

# with open('mcom_buffer_0____starting_session.txt', 'r') as f:
#     lines = f.readlines()

# pointer = 0
# cnt = 0
# while True:
#     if pointer+2 > len(lines): break
#     if "r [1.00,1.00,1.00]" in lines[pointer]:
#         lines[pointer+2] = ">>rec(%d, \"time\")\n"%cnt
#         cnt += 1
#     pointer += 1
    
# for i, line in enumerate(lines):
#     lines[i] = line.replace("w [", "w of=[")

# with open('x.txt', 'w+') as f:
#     f.writelines(lines)

#####################################################

# mcv = mcom(
#     path='./TEMP',
#     draw_mode='Img',
#     resume_mod=False,
#     image_path='./temp.jpg',
# )
# mcv.rec(1.1, 'x')
# mcv.rec(5, 'time')

# mcv.rec(1.2, 'x')
# mcv.rec(6, 'time')

# mcv.rec(1.3, 'x')
# mcv.rec(0.3, 'y')
# mcv.rec(7, 'time')

# mcv.rec(0.4, 'y')
# mcv.rec(1.4, 'x')
# mcv.rec(8, 'time')

# mcv.rec(0.5, 'y')
# mcv.rec(1.5, 'x')
# mcv.rec(9, 'time')
# mcv.rec_show()
input('wait complete')
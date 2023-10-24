import subprocess
# -v "mydinh3.MOV" -a annotation_Tung_mydinh3.txt
# subprocess.call(['python3','synopsis.py','-video','/home/sontung/Desktop/synopsis/mydinh3.mp4','-anno','/home/sontung/Desktop/synopsis/annotation_Tung_mydinh3.txt',
#                  '-overlap','1'])
# subprocess.call(['python3','tired.py','-video','vcc6.MOV','-anno','annotation_vcc6.txt','-overlap','0.5','-class','0'])
subprocess.call(['python3','synopsis_anno_mask.py','-video','centermask2/vcc6.MOV','-anno','centermask2/vcc6.npy','-overlap','1','-class','2'])


# subprocess.call(['python3','synopsis_shift.py','-video','/home/sontung/Desktop/synopsis/Hadong_ST.MOV','-anno','/home/sontung/Desktop/synopsis/Hadong_ST.txt',
#                  '-overlap','0.5','-start','2000','-finish','4000','-class','1'])

# subprocess.call(['python3','synopsis.py','-video','/home/sontung/Desktop/synopsis/vcc6.MOV','-anno','/home/sontung/Desktop/synopsis/annotation_vcc6.txt','-overlap','0.5',
#                  '-start','5400','-finish','7200'])
# subprocess.call(['python3','synopsis.py','-video','/home/sontung/Desktop/synopsis/vcc6.MOV','-anno','/home/sontung/Desktop/synopsis/vcc6_color_real.txt',
#                  '-overlap','1','-color','0','-class','1'])



# #Cach shift voi video mydinh3_lan2
# subprocess.call(['python3','synopsis_shift_1lane.py','-video','mydinh3.mp4','-anno','annotation_Tung_mydinh3.txt',
#                  '-overlap', '2','-numlane','2','-rotate','23','-deltax','-800'])
#
# # Cach shift voi video HaDong_lan2
# subprocess.call(['python3','synopsis_shift_1lane.py','-video','Hadong_ST.MOV','-anno','Hadong_ST.txt',
#                  '-overlap', '2','-numlane','2','-rotate','0','-deltax','-800'])
#
# # Cach shift voi video vcc6_lan2
# subprocess.call(['python3','synopsis_shift_1lane.py','-video','vcc6.MOV','-anno','annotation_vcc6.txt',
#                  '-overlap', '2','-numlane','2','-rotate','-5','-deltax','-800'])
#
# ##_____________________________________________________________________________________________________________________
#
#
# #Cach shift voi video mydinh3_lan1
# subprocess.call(['python3','synopsis_shift_1lane.py','-video','mydinh3.mp4','-anno','annotation_Tung_mydinh3.txt',
#                  '-overlap', '2','-numlane','1','-rotate','23','-deltax','-800'])
#
# #Cach shift voi video HaDong_lan1
# subprocess.call(['python3','synopsis_shift_1lane.py','-video','Hadong_ST.MOV','-anno','Hadong_ST.txt',
#                  '-overlap', '2','-numlane','1','-rotate','0','-deltax','-800'])
#
# # Cach shift voi video vcc6_lan1
# subprocess.call(['python3','synopsis_shift_1lane.py','-video','vcc6.MOV','-anno','annotation_vcc6.txt',
#                  '-overlap', '2','-numlane','1','-rotate','-5','-deltax','-800'])
#
# ##_____________________________________________________________________________________________________________________
#
# subprocess.call(['python3','synopsis.py','-video','mydinh3.mp4','-anno','annotation_Tung_mydinh3.txt',
#                  '-overlap','0.7'])
# # #Cach shift voi video mydinh3_thanhbeti
# subprocess.call(['python3','synopsis_2.py','-video','mydinh3.mp4','-anno','annotation_Tung_mydinh3.txt',
#                  '-overlap', '0.7'])

# #Cach shift voi video HaDong_thanhthunho
# subprocess.call(['python3','synopsis_2.py','-video','Hadong_ST.MOV','-anno','Hadong_ST.txt',
#                  '-overlap', '1.75'])
#
# #Cach shift voi video vcc6_thanhnhoxiu
# subprocess.call(['python3','synopsis_2.py','-video','vcc6.MOV','-anno','annotation_vcc6.txt',
#                  '-overlap', '1.75'])


# subprocess.call(['python3','synopsis_mask.py','-video','vcc6.MOV','-anno','annotation_vcc6.txt',
#                  '-overlap','1','-class','0'])
#
# subprocess.call(['python3','synopsis.py','-video','vcc6.MOV','-anno','annotation_vcc6.txt',
#                  '-overlap','1','-class','0'])

# subprocess.call(['python3','synopsis_shift_1lane.py','-video','mydinh3.mp4','-anno','annotation_Tung_mydinh3.txt',
#                  '-overlap', '1','-numlane','2','-rotate','0','-deltax','100'])
# subprocess.call(['python3','synopsis_2.py','-video','Hadong_ST.MOV','-anno','Hadong_ST.txt',
#                  '-overlap', '1.75'])
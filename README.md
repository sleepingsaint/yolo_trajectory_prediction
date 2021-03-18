# Trajectory
clone this repo using the following command
git clone  https://github.com/nithinvenny07/Trajectory 

download objection detection weights from the following link and save it in detector/YOLOV3/weight 
https://drive.google.com/file/d/10NPVgc8ExXradnEg5CNBHCqmeMxJugCO/view?usp=sharing 

download weights for trajectory prediction of arm from the following link and save it in Trajectory/individual/ 
https://drive.google.com/file/d/1VfRVvc-7EowI540S0_6FxJOMVg9XyXef/view?usp=sharing 

download weights for trajectory prediction of end effector from the following link and save it in Trajectory/individual/ 
https://drive.google.com/file/d/1t_qok3BNNHN6EK_Uw3WiXrfGtLfQJpCs/view?usp=sharing 

download weights for trajectory prediction of probe holder from the following link and save it in Trajectory/individual/ 
https://drive.google.com/file/d/1SEZGUvLB2gfVwA-yGTBlF5k3YDiOzYWj/view?usp=sharing 

download weights for trajectory prediction of person from the following link and save it in Trajectory/individual/eth_train/ 
https://drive.google.com/file/d/1p8vo9rig9Q0i0WLVudQBgzkdjtJXcDiS/view?usp=sharing

download weights for trajectory prediction of probe from the following link and save it in Trajectory/individual/
https://drive.google.com/file/d/1Lwsa12AP3K8KgwRpdCHR6o8tpAopSKLb/view?usp=sharing 
 
download link for sample video used in the code
https://drive.google.com/file/d/1wUvJfLrt8YihDDkMTG7E606izG9V5d4H/view?usp=sharing

run yolov3_deepsort.py file to get the results 
you can change the path of the video in this line https://github.com/nithinvenny07/Trajectory/blob/cafcd760c9fd0082661786a80071e8f3137cb934/yolov3_deepsort.py#L237 
alternatively you can run the following command in the terminal
python yolov3_deepsort.py --VIDEO_PATH "video path"

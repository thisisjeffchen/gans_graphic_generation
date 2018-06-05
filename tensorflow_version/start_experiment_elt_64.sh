#!/bin/bash 

#First argument is number of gen_updates
#Second argument is category
#Third argument is train image dir processed_segmented_cropped/processed_segmented
#Last argument is 
#Example: ./start_experiment.sh 5 elephant processed_segmented_cropped
#screen -S gen_updates_5
#screen -S 2329 -X sessionname my_session

GEN_UPDATES=2
CAT=$2
IMAGEDIR="processed_segmented_cropped"
DATE=`date +%Y%m%d`

python3 prep_data.py --experiment=${DATE}_ELT_${IMAGEDIR}_07_epochs_1000_batch_size_256_gen_updates_${GEN_UPDATES}_extra_64 --image_dir=Data/mscoco_raw/${IMAGEDIR}  --cat elephant laptop train
python3 train.py  --experiment=${DATE}_ELT_${IMAGEDIR}_07_epochs_1000_batch_size_256_gen_updates_${GEN_UPDATES}_extra_64 --epochs=1000 --gen_updates=${GEN_UPDATES} --extra_64 --image_dir=Data/mscoco_raw/${IMAGEDIR} --batch_size=256



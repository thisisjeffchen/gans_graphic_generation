#!/bin/bash 

#First argument is number of gen_updates
#Second argument is category
#Third argument is train image dir processed_segmented_cropped/processed_segmented
#Last argument is 
#Example: ./start_experiment.sh 2 elephant processed_segmented_cropped item_mafia
#screen -S gen_updates_5
#screen -S 2329 -X sessionname my_session

GEN_UPDATES=$1
CAT=$2
IMAGEDIR=$3
STYLEDIR=$4
DATE=`date +%Y%m%d`

python3 prep_data.py --experiment=${DATE}_${CAT}_${IMAGEDIR}_style_epochs_1000_batch_size_256_gen_updates_${GEN_UPDATES} --cat=${CAT}  --image_dir=Data/mscoco_raw/${IMAGEDIR}
python3 train.py  --experiment=${DATE}_${CAT}_${IMAGEDIR}_style_epochs_1000_batch_size_256_gen_updates_${GEN_UPDATES} --epochs=1000 --gen_updates=${GEN_UPDATES} --image_dir=Data/mscoco_raw/${IMAGEDIR} --style_image_dir=Data/style/${STYLEDIR} --batch_size=256



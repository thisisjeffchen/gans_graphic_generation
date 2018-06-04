#!/bin/bash 

#First argument is number of gen_updates
#Second argument is category
#Third argument is train image dir processed_segmented_cropped/processed_segmented
#uses 4800 as the thought vector length
#Example: ./start_experiment.sh 5 elephant processed_segmented_cropped
#screen -S gen_updates_5
#screen -S 2329 -X sessionname my_session

GEN_UPDATES=$1
CAT=$2
IMAGEDIR=$3
DATE=`date +%Y%m%d`

python3 prep_data.py --experiment=${DATE}_${CAT}_${IMAGEDIR}_07_epochs_1000_batch_size_256_gen_updates_${GEN_UPDATES}_caption_len_4800 --cat=${CAT}
python3 train.py  --experiment=${DATE}_${CAT}_${IMAGEDIR}_07_epochs_1000_batch_size_256_gen_updates_${GEN_UPDATES}_caption_len_4800 --epochs=1000 --gen_updates=${GEN_UPDATES} --image_dir=Data/mscoco_raw/${IMAGEDIR} --batch_size=256 --caption_vector_length=4800


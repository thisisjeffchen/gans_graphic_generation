#!/bin/bash
#Takes in experiment name as an input

EXP=$1

for EPOCH in 100 300 500 700 1000
do
    python3 generate_thought_vectors.py --experiment=${EXP}
    python3 generate_images.py --experiment=${EXP} --epoch=${EPOCH} --caption_vector_length=4800
    python3 copy_results.py --experiment=${EXP}
done


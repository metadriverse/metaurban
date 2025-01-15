#!/bin/bash
vila-infer \
    --model-path Efficient-Large-Model/VILA1.5-3b \
    --conv-mode vicuna_v1 \
    --text "Give me 2 numbers in the region of [-1, 1] as actions of the robot for the frame, the first dim is sterring and the second is the accelartion or deceleration, blue cubes in image indicate target position." \
    --media demo.png

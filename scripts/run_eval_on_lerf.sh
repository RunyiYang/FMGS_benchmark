#!/usr/bin/env bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#CONDA_PATH="/media/rocky/Fdisk/miniconda3"
#source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate fmgs

cd /media/rocky/Fdisk/Github/fmgs-opensource

eval_path=./data/tidy_lerf/fmgs_postprocessed_lerfdata_trainedweights/Localization_eval_puremycolmap
data_path=./data/tidy_lerf/fmgs_postprocessed_lerfdata_trainedweights
ckpts_path=./data/tidy_lerf/fmgs_postprocessed_lerfdata_trainedweights/new_ckpts_onlerf  #ckpts

iterations=(
34200
)


sequences=(
"bouquet"  
#"figurines"
#"ramen"
#"teatime"
#"waldo_kitchen"
)


for iter in "${!iterations[@]}"; do
echo "========================================"
echo "================ Eval for iteration ${iterations[iter]} ================="

for i in "${!sequences[@]}"; do

echo "========================================"
sequence_folder="$data_path/${sequences[i]}/${sequences[i]}"
echo "Running eval on: ${sequence_folder} ...... "


python ./render_lerf_relavancy_eval.py -s $sequence_folder -m $ckpts_path/${sequences[i]}  --dataformat colmap  --eval_keyframe_path_filename $eval_path/${sequences[i]}/keyframes_reversed_transform2colmap.json --iteration ${iterations[iter]} 

# Check the generated results saved in $ckpts_path/${sequences[i]}/test

done

done





 

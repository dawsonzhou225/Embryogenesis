#!/bin/bash
# run_suite.sh -- the perturbation suite: 12 configurations x 5 seeds, 256 steps, then the figure and GIFs.
# Every event hits blanket 0, the blanket under the organ's core.  ~28 min a run, 3 at once: ~9.5 h, ~17 GB each.
#   bash run_suite.sh 2>&1 | tee suite_log.txt
set -u
cd "$(dirname "$0")"
out=results
C="--seeds 0-4 --cores 3 --nY 256 --out $out"
run() { python3 run.py run "$@"; }

run --name base          $C
# the formed organ, step 150
run --name mut_flip      $C --mutate-at 150 --mutate-n 3 --mutate-what flip      # 3 ring cells' kind belief flipped
run --name mut_hold      $C --mutate-at 150 --mutate-n 3 --mutate-what hold      # ... and held by a prior for life
run --name abl_sen4      $C --ablate-at 150 --ablate-what sensory --ablate-n 4   # 4 sensory cells removed
run --name abl_core      $C --ablate-at 150 --ablate-what core                   # the core removed
run --name abl_blanket2  $C --organ-ablate-at 150 --organ-ablate-n 2             # 2 whole ensembles removed
run --name wall_formed   $C --wall-at 150 --wall-until 200                       # a signalling wall through the core
# during growth, step 60 (the organ has its 16 ensembles, ~100 of 256 cells)
run --name dev_hold      $C --mutate-at 60 --mutate-n 3 --mutate-what hold
run --name dev_core      $C --ablate-at 60 --ablate-what core
run --name dev_blanket2  $C --organ-ablate-at 60 --organ-ablate-n 2
run --name wall_dev      $C --wall-at 20 --wall-until 100
run --name wall_late     $C --wall-at 60 --wall-until 140

python3 run.py report --out $out | tee $out/report.txt
python3 fig.py --out $out
python3 gif.py --suite $out

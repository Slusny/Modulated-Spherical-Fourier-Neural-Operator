python main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/solar-spaceship-111-sID{0005} --no-shuffle

# jolly-blaze-113 (1film)
python main.py --save-forecast --model sfno --film-gen mae --model-version film --resume-checkpoint /mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578}/checkpoint_sfno_film_mae_iter=0_epoch=1.pkl --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/jolly-blaze-113-sID{13578} --num-iterations 73 --batch-size 5


# Sfno
python main.py --save-forecast --model sfno --model-version latest --multi-step-validation 4 --validation-step-skip 27 --output-path /mnt/qb/work2/goswami0/gkd965/checkpoints/sfno  --num-iterations 73 --batch-size 5
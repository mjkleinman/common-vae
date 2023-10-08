# Aug 5, 2023 (server - uswest 2 )
#for klqq in 0 1 5 10 50
#do
#python main_eval.py --name cvae_dshapes_randSample_klqq=${klqq}_klu=10_epoch=70_batch=128_z=9_zu=3_seed=0 --dataset dshapes --nu 3 --nz 9 --num-factors 6
#done

for klu in 10 50
do
  for klqq in 1 5 10
    do
    python main_eval.py --name cvae_dshapes_randSample_RegAnneal=100000_klqq=${klqq}_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=0 \
    --nu 3 --nz 9 --num-factors 6 --dataset dshapes
    done
done
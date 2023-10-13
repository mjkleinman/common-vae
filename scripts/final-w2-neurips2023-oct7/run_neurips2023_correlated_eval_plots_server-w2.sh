# Oct 8, 2023 (server - uswest 2 )
#for seed in 0 1 2
#do
#  for klu in 25 50 # 10
#  do
#    python main_eval.py --name cvae_dshapescorr_randSample_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#    --nu 3 --nz 9 --num-factors 6 --dataset dshapescorr
#  done
#done

#for seed in 2337
#do
#  for klu in 10
#  do
#    python main_eval.py --name cvae_dshapescorr_randSample_PERVIEWRECON_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#    --nu 3 --nz 9 --num-factors 6 --dataset dshapescorr
#  done
#done

# Oct 10, reruning with decoder from subset
#for seed in 0 1 2
#do
#  python main_eval.py --name cvae_dshapes_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#  --nu 3 --nz 9 --num-factors 6 --dataset dshapes
#done

for seed in 0 1 2
do
    python main_eval.py --name cvae_ddsprites2_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=7_zu=2_seed=${seed} \
      --nu 2 --nz 7 --num-factors 5 --dataset ddsprites2
done
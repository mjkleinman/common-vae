# Oct 12, Running on Correlated data with different decoders (parition or full)
for seed in 2 #0 1 2
do
  for dec in 'Doubleburgess' 'Doubleburgessindeprecon'
  do
  python main_eval.py --name cvae_dshapescorr_randSample_dec=${dec}_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
  --nu 3 --nz 9 --num-factors 6 --dataset dshapes
  done
done


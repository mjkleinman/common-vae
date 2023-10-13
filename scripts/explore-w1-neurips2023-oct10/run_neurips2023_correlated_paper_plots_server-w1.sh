# Oct 12, Running on Correlated data with different decoders (parition or full)
for seed in 0 1 2
do
  for dec in 'Doubleburgess' 'Doubleburgessindeprecon'
  do
  python plot_disentanglement_hinton.py --result-dir results-paper --name \
  cvae_dshapescorr_randSample_dec=${dec}_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed}

  python main_viz.py cvae_dshapescorr_randSample_dec=${dec}_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
  traversals -r 9 --is-posterior

  python main_viz.py cvae_dshapescorr_randSample_dec=${dec}_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
  traversals -r 9
  done
done


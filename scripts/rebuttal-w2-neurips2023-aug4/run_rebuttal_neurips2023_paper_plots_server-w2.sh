# Aug 5, 2023 (server - uswest 2 )

#for klqq in 0 1 5 10 50
#do
#python plot_disentanglement_hinton.py --result-dir results-paper --name \
#cvae_dshapes_randSample_klqq=${klqq}_klu=10_epoch=70_batch=128_z=9_zu=3_seed=0
#
#python main_viz.py cvae_dshapes_randSample_klqq=${klqq}_klu=10_epoch=70_batch=128_z=9_zu=3_seed=0 \
#traversals -r 9 --is-posterior
#
#python main_viz.py cvae_dshapes_randSample_klqq=${klqq}_klu=10_epoch=70_batch=128_z=9_zu=3_seed=0 \
#traversals -r 9
#done

# Aug 6, 2023 (server - uswest 2 )
for klqq in 1 5 10
do
  for klu in 10 50
  do
    python plot_disentanglement_hinton.py --result-dir results-paper --name \
    cvae_dshapes_randSample_RegAnneal=100000_klqq=${klqq}_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=0

    python main_viz.py cvae_dshapes_randSample_RegAnneal=100000_klqq=${klqq}_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=0 \
    traversals -r 9 --is-posterior

    python main_viz.py cvae_dshapes_randSample_RegAnneal=100000_klqq=${klqq}_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=0 \
    traversals -r 9
  done
done
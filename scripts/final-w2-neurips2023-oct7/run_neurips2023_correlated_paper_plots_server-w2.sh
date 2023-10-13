# Oct 8, 2023 (server - uswest 2 )
#for seed in 0 1 2
#do
#  for klu in 25 50 # 10
#  do
#  python plot_disentanglement_hinton.py --result-dir results-paper --name \
#  cvae_dshapescorr_randSample_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed}
#
#  python main_viz.py cvae_dshapescorr_randSample_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#  traversals -r 9 --is-posterior
#
#  python main_viz.py cvae_dshapescorr_randSample_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#  traversals -r 9
#  done
#done

# Oct 9, 2023
#for seed in 2337
#do
#  for klu in 10
#  do
#    python plot_disentanglement_hinton.py --result-dir results-paper --name \
#    cvae_dshapescorr_randSample_PERVIEWRECON_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed}
#
#    python main_viz.py cvae_dshapescorr_randSample_PERVIEWRECON_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#    traversals -r 9 --is-posterior
#
#    python main_viz.py cvae_dshapescorr_randSample_PERVIEWRECON_klqq=0.1_klu=${klu}_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#    traversals -r 9
#  done
#done

#Oct 10, reruning with decoder from subset dshapes
#for seed in 0 1 2
#do
#  python plot_disentanglement_hinton.py --result-dir results-paper --name \
#  cvae_dshapes_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed}
#
#  python main_viz.py cvae_dshapes_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#  traversals -r 9 --is-posterior
#
#  python main_viz.py cvae_dshapes_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=9_zu=3_seed=${seed} \
#  traversals -r 9
#done

#Oct 12, reruning with decoder from subset DSPRITES
for seed in 0 1 2
do
    python plot_disentanglement_hinton.py --result-dir results-paper --name \
    cvae_ddsprites2_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=7_zu=2_seed=${seed} \

    python main_viz.py cvae_ddsprites2_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=7_zu=2_seed=${seed} \
    traversals -r 7 --is-posterior

    python main_viz.py cvae_ddsprites2_randSample_PERVIEWRECON_klqq=0.1_klu=10_epoch=70_batch=128_z=7_zu=2_seed=${seed} \
    traversals -r 7
done
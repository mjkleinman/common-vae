#!/bin/bash
# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

echo "Downloading small_norb."
if [[ ! -d "small_norb" ]]; then
  mkdir small_norb
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat" ]]; then
  wget -O small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
  gunzip small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat" ]]; then
  wget -O small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
  gunzip small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-training-info.mat" ]]; then
  wget -O small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
  gunzip small_norb/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat" ]]; then
  wget -O small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
  gunzip small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x46789x9x18x6x2x96x96-testing-cat.mat" ]]; then
  wget -O small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
  gunzip small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
fi
if [[ ! -e small_norb/"smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat" ]]; then
  wget -O small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
  gunzip small_norb/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
fi
echo "Downloading small_norb completed!"

echo "Downloading cars dataset."
if [[ ! -d "cars" ]]; then
  wget -O nips2015-analogy-data.tar.gz http://www.scottreed.info/files/nips2015-analogy-data.tar.gz
  tar xzf nips2015-analogy-data.tar.gz
  rm nips2015-analogy-data.tar.gz
  mv data/cars .
  rm -r data
fi
echo "Downloading cars completed!"


echo "Downloading scream picture."
if [[ ! -d "scream" ]]; then
  mkdir scream
  wget -O scream/scream.jpg https://upload.wikimedia.org/wikipedia/commons/f/f4/The_Scream.jpg
fi
echo "Downloading scream completed!"

echo "Downloading mpi3d_toy dataset."
if [[ ! -d "mpi3d_toy" ]]; then
  mkdir mpi3d_toy
  wget -O mpi3d_toy/mpi3d_toy.npz https://storage.googleapis.com/disentanglement_dataset/data_npz/sim_toy_64x_ordered_without_heldout_factors.npz
fi
echo "Downloading mpi3d_toy completed!"

echo "Downloading 3dshapes dataset."
if [[ ! -d "shapes3d" ]]; then
  mkdir shapes3d
  wget -O shapes3d/3dshapes.h5 https://storage.googleapis.com/3d-shapes/3dshapes.h5
fi
echo "Downloading 3dshapes completed!"

#!/bin/bash

# create conda environment
eval "$(conda shell.bash hook)"
conda activate envpool-dev
if [ $? -ne 0 ]; then
    conda create -n envpool-dev python=3.10 -y
    conda activate envpool-dev
fi

ENVPOOL_NAME=minimal-envpool
ENV_NAME=gobang_mcts

# clone envpool
rm -rf ${ENVPOOL_NAME}
git clone git@github.com:CWHer/minimal-envpool.git
ln -s ${PWD}/${ENV_NAME} ${ENVPOOL_NAME}/envpool/${ENV_NAME}
cd ${ENVPOOL_NAME}

# apply patches
git apply ../patches/envpool.patch

# build all
make bazel-clean
make bazel-test
# generate .whl file
make bazel-release
# install .whl
pip install dist/* --force-reinstall

cd ..
pip install tqdm
python -m unittest gobang_envpool_test.py
if [ $? -ne 0 ]; then
    echo "Test failed!"
    exit 1
fi

conda deactivate

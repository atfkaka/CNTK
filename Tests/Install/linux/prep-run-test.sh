#!/bin/bash
# TODO nvidia-smi to check availability of GPUs for GPU tests

CNTK_DROP=\$HOME/cntk

RUN_TEST=/home/testuser/run-test.sh
cat >| $RUN_TEST <<RUNTEST
set -e -x
TEST_DEVICE=\$1
export PATH="$PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
source ~/cntk/activate-cntk

# Just for informational purposes:
[ "\$TEST_DEVICE" = "gpu" ] && nvidia-smi

TEST_DEVICE_ID=-1
[ "\$TEST_DEVICE" = "gpu" ] && TEST_DEVICE_ID=0

which cntk
MODULE_DIR="\$(python -c "import cntk, os, sys; sys.stdout.write(os.path.dirname(os.path.abspath(cntk.__file__)))")"
[ \$? -eq 0 ]

[ "\$TEST_DEVICE" = "gpu" ] && pytest "\$MODULE_DIR" --deviceid \$TEST_DEVICE --doctest-modules
# TODO not all (doc) tests run on CPU:
#=================================== FAILURES ===================================
#____________________ [doctest] cntk.ops.optimized_rnnstack _____________________
#069         recurrent_op (str, optional): one of 'lstm', 'gru', 'relu', or 'tanh'.
#070         name (str, optional): the name of the Function instance in the network
#071
#072     Example:
#073         >>> from _cntk_py import InferredDimension, constant_initializer
#074         >>> W = C.parameter((InferredDimension,4), constant_initializer(0.1))
#075         >>> x = C.input_variable(shape=(4,))
#076         >>> s = np.reshape(np.arange(20.0, dtype=np.float32), (5,4))
#077         >>> f = C.optimized_rnnstack(x, W, 8, 2)
#078         >>> f.eval({x:s}).shape
#UNEXPECTED EXCEPTION: RuntimeError('Inside File: Source/Math/Matrix.cpp  Line: 4415  Function: RNNForward  -> Feature Not Implemented.',)
#Traceback (most recent call last):
#
#  File "/home/testuser/anaconda3/envs/cntk-py34/lib/python3.4/doctest.py", line 1318, in __run
#    compileflags, 1), test.globs)
#
#  File "<doctest cntk.ops.optimized_rnnstack[5]>", line 1, in <module>
#
#  File "/home/testuser/anaconda3/envs/cntk-py34/lib/python3.4/site-packages/cntk/ops/functions.py", line 179, in eval
#    _, output_map = self.forward(arguments, self.outputs, device=device)
#
#  File "/home/testuser/anaconda3/envs/cntk-py34/lib/python3.4/site-packages/cntk/utils/swig_helper.py", line 58, in wrapper
#    result = f(*args, **kwds)
#
#  File "/home/testuser/anaconda3/envs/cntk-py34/lib/python3.4/site-packages/cntk/ops/functions.py", line 239, in forward
#    keep_for_backward)
#
#  File "/home/testuser/anaconda3/envs/cntk-py34/lib/python3.4/site-packages/cntk/cntk_py.py", line 1137, in _forward
#    return _cntk_py.Function__forward(self, *args)
#
#RuntimeError: Inside File: Source/Math/Matrix.cpp  Line: 4415  Function: RNNForward  -> Feature Not Implemented.
#
#/home/testuser/anaconda3/envs/cntk-py34/lib/python3.4/site-packages/cntk/ops/__init__.py:78: UnexpectedException
#----------------------------- Captured stderr call -----------------------------
#Inside File: Source/Math/Matrix.cpp  Line: 4415  Function: RNNForward  -> Feature Not Implemented.
#===================== 1 failed, 552 passed in 6.92 seconds =====================


# Installation validation example from CNTK.wiki (try from two different paths):
cd "$CNTK_DROP/Tutorials"

python NumpyInterop/FeedForwardNet.py
cd NumpyInterop
python FeedForwardNet.py

cd "$CNTK_DROP/Examples/Image/DataSets/MNIST"
python install_mnist.py

cd "$CNTK_DROP/Examples/Image/DataSets/CIFAR-10"
python install_cifar10.py

# TODO run some examples

# TODO actually do different device and syntax.

if [ "\$TEST_DEVICE" = "gpu" ]; then
  cd "$CNTK_DROP/Tutorials"
  for f in *.ipynb; do
    # TODO 203 fails when run without GUI?
    if [ "\$f" != "CNTK_203_Reinforcement_Learning_Basics.ipynb" ]; then
      jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1200 --output \$(basename \$f .ipynb)-out.ipynb \$f
    fi
  done
fi

# CNTK.wiki example:
cd "$CNTK_DROP/Tutorials/HelloWorld-LogisticRegression"
cntk configFile=lr_bs.cntk deviceId=\$TEST_DEVICE_ID

cd "$CNTK_DROP/Examples/Image/GettingStarted"
cntk configFile=01_OneHidden.cntk deviceId=\$TEST_DEVICE_ID

RUNTEST

chmod 755 $RUN_TEST
chown testuser:testuser $RUN_TEST

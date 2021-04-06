bazel build --config=cuda //tensorflow/tools/pip_package:build_pip_package && \
  ./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg && \
  pip3 uninstall -y tf-nightly && \
  pip3 install /tmp/tensorflow_pkg/tf_nightly-2.6.0-cp38-cp38-linux_x86_64.whl && \
  pip3 install --upgrade numpy

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // diff_i=a_i-b_i
  const int channels = bottom[0]->channels();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    Dtype dot;
    caffe_gpu_dot(channels,
        diff_.gpu_data() + (i*channels), diff_.gpu_data() + (i*channels), &dot);
    top[0]->mutable_cpu_data()[i]=dot; 
  }
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanDistLayer);

}  // namespace caffe

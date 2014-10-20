#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // diff_i=a_i-b_i
  const int channels = bottom[0]->channels();
  //Dtype margin = this->layer_param_.euclidean_dist_param().margin();
  //Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    Dtype dot;
    caffe_gpu_dot(channels,
        diff_.gpu_data() + (i*channels), diff_.gpu_data() + (i*channels), &dot);
    (*top)[0]->mutable_cpu_data()[i]=dot; 
    //dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        //diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    //if (static_cast<int>(bottom[2]->cpu_data()[i])) {  // similar pairs
    //  loss += dist_sq_.cpu_data()[i];
    //} else {  // dissimilar pairs
    //  loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
    //}
  }
  //loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  //(*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanDistLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(EuclideanDistLayer);

}  // namespace caffe

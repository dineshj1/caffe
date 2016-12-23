#ifndef CAFFE_EUCLIDEAN_DIST_LAYER_HPP_
#define CAFFE_EUCLIDEAN_DIST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the euclidean distance@f$
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times 1 \times 1) @f$
 *      the features @f$ a \in [-\infty, +\infty]@f$
 *   -# @f$ (N \times C \times 1 \times 1) @f$
 *      the features @f$ b \in [-\infty, +\infty]@f$
 * @param top output Blob vector (length 1)
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 */
template <typename Dtype>
class EuclideanDistLayer : public Layer<Dtype> {
 public:
  explicit EuclideanDistLayer(const LayerParameter& param)
      : Layer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline const char* type() const { return "EuclideanDist"; }

 protected:
  /// @copydoc EuclideanDistLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;  // cached for backward pass
  //Blob<Dtype> dist_sq_;  // cached for backward pass
  //Blob<Dtype> diff_sq_;
  //Blob<Dtype> summer_vec_;
};

}  // namespace caffe

#endif  // CAFFE_EUCLIDEAN_DIST_LAYER_HPP_

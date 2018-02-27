#ifndef CAFFE_ALGEBRA_LAYERS_HPP_
#define CAFFE_ALGEBRA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief MatMultLayer. Compute C = AxB. It handles the case where either A or B are full or diagonal.
 */
template <typename Dtype>
class MatMultLayer : public Layer<Dtype> {
 public:
  virtual inline const char* type() const { return "MatMult"; }
  explicit MatMultLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
      
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  // Compute height_out_ and width_out_ from other parameters.
//  virtual void compute_output_shape() = 0;

 private:
  /// @brief The spatial dimensions of the first input.
  vector<int> a_shape_;
  /// @brief The spatial dimensions of the second input.
  vector<int> b_shape_;

  int N_M_;
  int A_offset_;
  int B_offset_;
  int C_offset_;
  int D_1_;
  int D_2_;
  int D_3_;
  bool A_is_diag_;
  bool B_is_diag_;
  CBLAS_TRANSPOSE A_transpose_;
  CBLAS_TRANSPOSE B_transpose_;
};


/**
 * @brief MatInvLayer. Compute A = (A + \lambda I)^{-1}.
 */
template <typename Dtype>
class MatInvLayer : public Layer<Dtype> {
 public:
 virtual inline const char* type() const { return "MatInv"; }
 
  explicit MatInvLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> tmp_buffer_shape_;
 
 private:
  float lambda_;
  vector<int> input_shape_;
  int N_;
  int dim_;
  int offset_;
  Blob<Dtype> tmp_buffer_; 
};

}  // namespace caffe

#endif  // CAFFE_ALGEBRA_LAYERS_HPP_

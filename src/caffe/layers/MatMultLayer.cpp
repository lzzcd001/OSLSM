#include <vector>
#include "caffe/algebra_layers.hpp"
namespace caffe {

template <typename Dtype>
void MatMultLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {    
  // See if A or B are diagonal
  MatMultParameter matmult_param = this->layer_param_.matmult_param();
  int field_size = matmult_param.diagonal_input_size();
  A_is_diag_ = (field_size > 0 && matmult_param.diagonal_input(0));
  B_is_diag_ = (field_size > 1 && matmult_param.diagonal_input(1));
	
  // Does not work if A is full and B is diagonal
  CHECK(A_is_diag_ || !B_is_diag_) << "Full A times diagonal B is not supported.";
	
  // See if we need to transpose A and B
  A_transpose_ = matmult_param.transpose_a() ? CblasTrans : CblasNoTrans;
  B_transpose_ = matmult_param.transpose_b() ? CblasTrans : CblasNoTrans;
}

template <typename Dtype>
void MatMultLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  a_shape_ = bottom[0]->shape();
  b_shape_ = bottom[1]->shape();
  //LOG(ERROR) << this->layer_param_.name();
  //LOG(ERROR) << a_shape_[0] << ", " << a_shape_[1] << ", "  << a_shape_[2] << ", "  << a_shape_[3];
  //LOG(ERROR) << b_shape_[0] << ", "  << b_shape_[1] << ", "  << b_shape_[2] << ", "  << b_shape_[3];
  
  int a_start_axis = 0;
  int b_start_axis = 0;

  if(A_is_diag_) { 
    CHECK_GE(a_shape_.size(), 1);
    a_start_axis = a_shape_.size() - 1;
    D_1_ = a_shape_[a_start_axis];
    D_2_ = D_1_;
    A_offset_ = D_1_;
  } else {
    CHECK_GE(a_shape_.size(), 2);
    a_start_axis = a_shape_.size() - 2;
    if(A_transpose_ == CblasTrans) {
      D_1_ = a_shape_[a_start_axis + 1];
      D_2_ = a_shape_[a_start_axis];
    } else {
      D_1_ = a_shape_[a_start_axis];
      D_2_ = a_shape_[a_start_axis + 1];
    }
      A_offset_ = D_1_ * D_2_;
  }
	
  if(B_is_diag_) { 
    CHECK_GE(b_shape_.size(), 1);
    b_start_axis = b_shape_.size() - 1;
    CHECK_EQ(D_2_, b_shape_[b_start_axis]) << "Matrices dimension do not match";	
    D_3_ = D_2_;
    B_offset_ = D_2_;
  } else {
    CHECK_GE(a_shape_.size(), 2);
    b_start_axis = b_shape_.size() - 2;
    
    if(B_transpose_ == CblasTrans) {
      CHECK_EQ(D_2_, b_shape_[b_start_axis + 1]) << "Matrices dimension do not match";
      D_3_ = b_shape_[b_start_axis];
    } else {
      CHECK_EQ(D_2_, b_shape_[b_start_axis]) << "Matrices dimension do not match";
      D_3_ = b_shape_[b_start_axis + 1];
    }
      B_offset_ = D_2_ * D_3_;
  }
	
  N_M_ = bottom[0]->count(0, a_start_axis);
  CHECK_EQ(N_M_, bottom[1]->count(0, b_start_axis)) << "Num of Matrices should be the same";
	
	
  vector<int> c_shape;
  c_shape.insert(c_shape.end(), a_shape_.begin(), a_shape_.begin() + a_start_axis); 
  c_shape.push_back(D_1_);
	
	
  if(A_is_diag_ && B_is_diag_) {
    C_offset_ = D_1_; 
  } else {
    C_offset_ = D_1_ * D_3_;
    c_shape.push_back(D_3_);
  }
  top[0]->Reshape(c_shape); 
  
  //LOG(ERROR) << this->layer_param_.name();
  //LOG(ERROR) << c_shape[0] << ", " << c_shape[1] << ", "  << c_shape[2] << ", "  << c_shape[3];
}

template <typename Dtype>
void MatMultLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
}


template <typename Dtype>
void MatMultLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MatMultLayer);
#endif


INSTANTIATE_CLASS(MatMultLayer);
REGISTER_LAYER_CLASS(MatMult);

}  // namespace caffe

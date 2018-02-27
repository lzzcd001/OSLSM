#include <vector>

#include "caffe/algebra_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ScaleMatRow(const int nthreads,
    const Dtype* input_mat, const Dtype* scales, Dtype* output_mat, 
    	int lda) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	int row_index = index / lda;
  	output_mat[index] = input_mat[index] * scales[row_index];
  }
}

template <typename Dtype>
__global__ void MatMatDot(const int nthreads,
    const Dtype* mat_a, const Dtype* mat_b, Dtype* output_data, 
    	int lda) {
  CUDA_KERNEL_LOOP(index, nthreads) {
  	output_data[index] = 0;
  	int offset = index * lda;
  	for(int i = offset; i < (offset + lda); ++i) {
  	  	output_data[index] += mat_a[i] * mat_b[i];
  	}
  }
}
  

template <typename Dtype>
void MatMultLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	
	const Dtype* A_data = bottom[0]->gpu_data();
	const Dtype* B_data = bottom[1]->gpu_data();
	Dtype* C_data = top[0]->mutable_gpu_data();
	
	//handle the case both A and B are full matrices: C = AB
	if(!A_is_diag_ && !B_is_diag_) {
		for (int n = 0; n < N_M_; ++n) {
			caffe_gpu_gemm<Dtype>(A_transpose_, B_transpose_, D_1_,
    	    	D_3_, D_2_,
    	    	(Dtype)1., A_data + A_offset_ * n, B_data + B_offset_ * n,
    	    	(Dtype)0., C_data + C_offset_ * n);
		}
	}
	//if A is diagonal we scale each row of B by 
	//coressponding coefficient in diagonal of A
	else if(A_is_diag_ && !B_is_diag_) {
	    if(B_transpose_ == CblasNoTrans) {
		    int count = top[0]->count();
		    ScaleMatRow<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                count, B_data, A_data, C_data, D_3_);
		} else {
		    LOG(FATAL) << "B can not be transposed while A is diagonal!";    
		}
	} else if(!A_is_diag_ && B_is_diag_) {
		LOG(FATAL) << "B can not be diagonal while A is not diagonal!";		
	} else {
		caffe_gpu_mul(N_M_ * C_offset_, A_data, B_data, C_data);		
	}
}


template <typename Dtype>
void MatMultLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	
	const Dtype* A_data = bottom[0]->gpu_data();

	const Dtype* B_data = bottom[1]->gpu_data();

    const Dtype* C_diff = top[0]->gpu_diff();
	Dtype* A_diff = bottom[0]->mutable_gpu_diff();
	Dtype* B_diff = bottom[1]->mutable_gpu_diff();	
	
	//Both A and B are full matrices: 
	//A' = C' B^\top
	//B' = A^\top C'
	if(!A_is_diag_ && !B_is_diag_) {
		if(A_transpose_ == CblasNoTrans && B_transpose_ == CblasNoTrans) {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, D_1_,
						D_2_, D_3_, (Dtype)1., 
						C_diff + C_offset_ * n, B_data + B_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, D_2_,
						D_3_, D_1_, (Dtype)1., 
						A_data + A_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., B_diff + B_offset_ * n);
		    	}
		    }
		} else if(A_transpose_ == CblasTrans && B_transpose_ == CblasNoTrans) {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, D_2_,
						D_1_, D_3_,
						(Dtype)1., B_data + B_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, D_2_,
						D_3_, D_1_,
						(Dtype)1., A_data + A_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., B_diff + B_offset_ * n);
				}
		    }
		} else if(A_transpose_ == CblasNoTrans && B_transpose_ == CblasTrans) {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, D_1_,
						D_2_, D_3_, (Dtype)1., 
						C_diff + C_offset_ * n, B_data + B_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, D_3_,
						D_2_, D_1_, (Dtype)1., 
						C_diff + C_offset_ * n, A_data + A_offset_ * n,
						(Dtype)0., B_diff + B_offset_ * n);
		    	}
		    }
		} else if(A_transpose_ == CblasTrans && B_transpose_ == CblasTrans) {
			for (int n = 0; n < N_M_; ++n) {
				if (propagate_down[0]) {
					caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans, D_2_,
						D_1_, D_3_,
						(Dtype)1., B_data + B_offset_ * n, C_diff + C_offset_ * n,
						(Dtype)0., A_diff + A_offset_ * n);
				}
				if (propagate_down[1]) {
					caffe_gpu_gemm<Dtype>(CblasTrans, CblasTrans, D_3_,
						D_2_, D_1_, (Dtype)1., 
						C_diff + C_offset_ * n, A_data + A_offset_ * n, 
						(Dtype)0., B_diff + B_offset_ * n);
				}
		    }
		}		
	}
	else if(A_is_diag_ && !B_is_diag_) {
		if (propagate_down[1]) {
			int count = top[0]->count();
			ScaleMatRow<Dtype><<<CAFFE_GET_BLOCKS(count), 
			CAFFE_CUDA_NUM_THREADS>>>(count, C_diff, A_data, B_diff, D_3_);  
		}
		if (propagate_down[0]) {
			int row_count = bottom[0]->count();
			MatMatDot<Dtype><<<CAFFE_GET_BLOCKS(row_count), 
				CAFFE_CUDA_NUM_THREADS>>>(row_count, C_diff, B_data, A_diff, D_3_);
		}
	} else if(!A_is_diag_ && B_is_diag_) {
		LOG(FATAL) << "B can not be diagonal while A is not diagonal!";	
	} else {
		if (propagate_down[0]) {
			caffe_gpu_mul(N_M_ * C_offset_, C_diff, B_data, A_diff);
		}
		if (propagate_down[1]) {
			caffe_gpu_mul(N_M_ * C_offset_, A_data, C_diff, B_diff);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(MatMultLayer);

}  // namespace caffe

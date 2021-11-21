CC=gcc
CXX=g++
NVCC=nvcc
COMMON_DIR=../common

#CUDA code generation flags
#GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
#GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
#GENCODE_FLAGS   :=  $(GENCODE_SM30) $(GENCODE_SM35)

#OPTIONS_CUDA   := -O4 -ftz=false  $(GENCODE_FLAGS) -DCUDA -DCUDADEBUG -DCOUNT -DDATATYPEDOUBLE
OPTIONS_CUDA   := -O4 -ftz=false -G -DCUDA -DCUDADEBUG -DCOUNT -DDATATYPEDOUBLE


CUDAVARIANTS = 7_reduce_smem_complete_unrolling\
	       7_reduction_smem

all: $(CUDAVARIANTS) $(CVARIANTS)

cuda: $(CUDAVARIANTS)

$(CUDAVARIANTS): % : %.cu $(COMMON_DIR)/common.cpp $(COMMON_DIR)/cuda_common.cu 
		$(NVCC) $^ -o  $@ $(OPTIONS_CUDA)

clean:
	rm $(CUDAVARIANTS)

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>
int main(void) {
  size_t sz = 2UL << 20;
  void *p = NULL;
  cudaSetDevice(0);
  if (cudaMalloc(&p, sz) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
  printf("cudaMalloc: ok  ptr=%p\n", p);

  int fd = -1;
  CUresult r = cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)p, sz,
        CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  printf("cuMemGetHandleForAddressRange(DMA_BUF_FD): %d  fd=%d\n", (int)r, fd);

  int n = 0; struct ibv_device **list = ibv_get_device_list(&n);
  printf("ib devices: %d\n", n);
  if (!n) return 3;
  struct ibv_context *c = ibv_open_device(list[0]);
  struct ibv_pd *pd = ibv_alloc_pd(c);

  if (r == CUDA_SUCCESS) {
    struct ibv_mr *mr = ibv_reg_dmabuf_mr(pd, 0, sz, (uint64_t)p, fd,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    printf("ibv_reg_dmabuf_mr  on %s: %s (errno=%d)\n", ibv_get_device_name(list[0]),
           mr ? "SUCCESS" : "FAILED", mr ? 0 : errno);
  }
  struct ibv_mr *m2 = ibv_reg_mr(pd, p, sz,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
  printf("ibv_reg_mr (peermem) on %s: %s (errno=%d)\n", ibv_get_device_name(list[0]),
         m2 ? "SUCCESS" : "FAILED", m2 ? 0 : errno);
  return 0;
}

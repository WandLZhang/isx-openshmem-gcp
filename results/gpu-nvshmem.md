# GPU: NVSHMEM

Two machines. `a2-highgpu-2g` (2 x A100, NVLink) and `a3-ultragpu-8g` (8 x H200, NVLink
plus 8 x ConnectX-7 RoCE NICs) in us-east4-b.

## Single node

`src/gpu/isx_nvshmem.cu`, one PE per GPU, every run validated.

| GPUs | total | time | bucket | exchange | radix | rate |
|---:|---:|---:|---:|---:|---:|---:|
| 2 x A100 | 8.59 GB | 0.68 s | 80% | 8% | 10% | 12.61 GB/s |
| 8 x H200 | 2.15 GB | 0.033 s | 0.026 | 0.004 | 0.002 | 65.05 GB/s |
| 8 x H200 | 8.59 GB | 0.130 s | 0.105 | 0.015 | 0.008 | 66.19 GB/s |
| 8 x H200 | 34.36 GB | 0.513 s | 0.422 | 0.055 | 0.033 | 66.96 GB/s |
| 8 x H200 | **137.44 GB** | 2.047 s | 1.686 | 0.215 | 0.131 | **67.15 GB/s** |

Rate holds at 67 GB/s across a 64x data range, so the implementation weak-scales inside a
node. Bucketing is 82% of runtime and the exchange over NVLink is 10%, the reverse of H4D
where the exchange is 89%.

Bucketing is a full `SortPairs` over (destination, key). The exchange needs keys grouped
rather than sorted within a group, so a partition would be cheaper. Correctness came
first, and this is the first thing to change if single-node time matters.

## Multi-node does not work with packaged NVSHMEM

Two `a3-ultragpu-8g`, 16 GPUs, on a VPC built from the `us-east4-b-vpc-roce` network
profile with 8 MRDMA NICs per node.

The fabric is fine. The GPU memory registration is not.

### The fabric works

`ib_write_bw` host to host, one NIC of eight, GID index 3:

```
 #bytes   #iterations  BW peak[MB/sec]  BW average[MB/sec]  MsgRate[Mpps]
 65536    5000         45368.13         45016.05            0.720257
```

45.0 GB/s, so 360 Gbps on one of eight NICs. All 8 HCAs report `PORT_ACTIVE` at MTU 4096.
One-sided RDMA Write across nodes works.

GID index matters. NVSHMEM defaults to index 0, which is RoCE v1 link-local (`fe80::`) and
cannot route across the RDMA subnet. Index 3 is RoCE v2 with the IPv4-mapped address.

| GID | type | address |
|---:|---|---|
| 0 | RoCE v1 | `fe80::182e:cfff:fea5:cb01` |
| 1 | RoCE v2 | `fe80::182e:cfff:fea5:cb01` |
| 2 | RoCE v1 | `::ffff:10.200.0.2` |
| **3** | **RoCE v2** | **`::ffff:10.200.0.2`** |

### Registering GPU memory is where it stops

A 30-line probe, `tools/dmabuf_reg_test.c`, registers the same `cudaMalloc` buffer two ways
on `rocep145s0`:

```
cudaMalloc: ok  ptr=0x7abbcbe00000
cuMemGetHandleForAddressRange(DMA_BUF_FD): 0  fd=64
ibv_reg_dmabuf_mr  on rocep145s0: SUCCESS (errno=0)
ibv_reg_mr (peermem) on rocep145s0: FAILED (errno=14)
```

The NIC registers GPU memory through dmabuf and refuses the peer-memory path.
`ibv_reg_mr` on a device pointer needs `nvidia_peermem`, and that module cannot load:

```
$ sudo modprobe nvidia_peermem
modprobe: ERROR: could not insert 'nvidia_peermem': Invalid argument
```

The module file is present at
`/lib/modules/6.8.0-1066-gcp/kernel/nvidia-580srv-open/nvidia-peermem.ko`. It fails to
insert because it registers against `ib_register_peer_memory_client`, an API that exists in
MOFED's `ib_core` and not in the inbox one this image ships.

Both NVSHMEM remote transports fail on this, for different reasons:

| transport | failure | why |
|---|---|---|
| `ibrc` | `ibv_poll_cq completion status 5`, then `progress_send failed` | registers the symmetric heap with `ibv_reg_mr`, which needs peermem |
| `ibgda` | `cudaHostRegister with IoMemory failed with error=800` | maps the NIC doorbell into GPU BAR space, which virtualized MRDMA does not expose |

The apt package `libnvshmem3-cuda-12` has no dmabuf setting at all: `nvshmem-info -a`
matches zero entries for `dmabuf`. So the one registration path the hardware accepts is the
one this build cannot ask for.

### Three ways to fix it

In order of effort.

1. **Build NVSHMEM from source with dmabuf registration.** The hardware side already works,
   proven above. This keeps the inbox driver stack and the standard image.
2. **Use a MOFED image.** MOFED's `ib_core` carries the peer-memory API, `nvidia_peermem`
   inserts, and stock `ibrc` then works unchanged.
3. **Route the exchange through NCCL/gIB.** Google qualifies NCCL with the gIB plugin on
   A3 Ultra and A4, and documents no NVSHMEM support on either. This is the supported path
   and the furthest from the OpenSHMEM one-sided model the study requires, so it is a last
   resort.

This repeats the H4D lesson. The provider the platform qualifies is the one that works, and
the stock default is not it.

## What this means for GB300

A4X Max has the same ConnectX RoCE fabric between NVL72 domains, so the same registration
question applies to any traffic that leaves a domain. Settle it before the allocation
starts: build the image, run `tools/dmabuf_reg_test.c`, and run a 2-node NVSHMEM job. That
is an hour of work on 2 nodes and it derisks the whole run.

Inside one NVL72 domain the transport is NVLink and none of this applies. The 8-GPU H200
numbers above are the closest available evidence for intra-domain behaviour.

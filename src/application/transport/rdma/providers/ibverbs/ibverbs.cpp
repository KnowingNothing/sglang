// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "mori/application/transport/rdma/providers/ibverbs/ibverbs.hpp"

#include <hip/hip_runtime_api.h>
#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>  // dereferences ibvHandle.qp/cq/srq (forward-declared in core)

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "mori/application/utils/check.hpp"
#include "mori/application/utils/math.hpp"
#include "mori/utils/mori_log.hpp"
namespace mori {
namespace application {

namespace {

class RdmaError : public std::runtime_error {
 public:
  RdmaError(int errnoValue, std::string what)
      : std::runtime_error(std::move(what)), errno_(errnoValue) {}

  int Errno() const noexcept { return errno_; }

 private:
  int errno_{0};
};

bool HasAnyNonZeroByte(const uint8_t* bytes, size_t len) {
  return std::any_of(bytes, bytes + len, [](uint8_t byte) { return byte != 0; });
}

std::string BytesToHex(const uint8_t* bytes, size_t len) {
  std::ostringstream os;
  os << std::hex << std::setfill('0');
  for (size_t i = 0; i < len; ++i) {
    if (i != 0) os << ':';
    os << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return os.str();
}

const char* LinkLayerName(uint8_t linkLayer) {
  switch (linkLayer) {
    case IBV_LINK_LAYER_UNSPECIFIED:
      return "unspecified";
    case IBV_LINK_LAYER_INFINIBAND:
      return "infiniband";
    case IBV_LINK_LAYER_ETHERNET:
      return "ethernet";
    default:
      return "unknown";
  }
}

bool UseIbverbsGpuDirect() {
  const char* value = std::getenv("MORI_IBVERBS_GPU_DIRECT");
  return value != nullptr && std::atoi(value) != 0;
}

std::string DescribeQpTransition(const char* transition, const ibv_qp_attr& attr, int flags,
                                 const ibv_device_attr_ex& devAttr, const ibv_port_attr& portAttr,
                                 const RdmaEndpointHandle& local, const char* devName) {
  std::ostringstream os;
  os << "transition=" << (transition != nullptr ? transition : "unknown")
     << " dev=" << (devName != nullptr ? devName : "unknown") << " flags=0x" << std::hex << flags
     << std::dec << " qp_state=" << static_cast<int>(attr.qp_state)
     << " port_num=" << static_cast<int>(attr.port_num)
     << " path_mtu=" << static_cast<int>(attr.path_mtu) << " dest_qp_num=" << attr.dest_qp_num
     << " rq_psn=" << attr.rq_psn << " sq_psn=" << attr.sq_psn
     << " max_dest_rd_atomic=" << static_cast<int>(attr.max_dest_rd_atomic)
     << " max_rd_atomic=" << static_cast<int>(attr.max_rd_atomic)
     << " min_rnr_timer=" << static_cast<int>(attr.min_rnr_timer)
     << " timeout=" << static_cast<int>(attr.timeout)
     << " retry_cnt=" << static_cast<int>(attr.retry_cnt)
     << " rnr_retry=" << static_cast<int>(attr.rnr_retry)
     << " ah_attr.is_global=" << static_cast<int>(attr.ah_attr.is_global)
     << " ah_attr.sl=" << static_cast<int>(attr.ah_attr.sl) << " ah_attr.dlid=" << attr.ah_attr.dlid
     << " ah_attr.port_num=" << static_cast<int>(attr.ah_attr.port_num)
     << " grh.sgid_index=" << static_cast<int>(attr.ah_attr.grh.sgid_index)
     << " grh.hop_limit=" << static_cast<int>(attr.ah_attr.grh.hop_limit)
     << " grh.traffic_class=" << static_cast<int>(attr.ah_attr.grh.traffic_class)
     << " local.qpn=" << local.qpn << " local.psn=" << local.psn << " local.portId=" << local.portId
     << " local.gidIdx=" << local.eth.gidIdx
     << " local.gid=" << BytesToHex(local.eth.gid, sizeof(local.eth.gid));

  if (HasAnyNonZeroByte(local.eth.mac, sizeof(local.eth.mac))) {
    os << " local.mac=" << BytesToHex(local.eth.mac, sizeof(local.eth.mac));
  }

  os << " caps.max_qp_rd_atom=" << devAttr.orig_attr.max_qp_rd_atom
     << " caps.max_qp_init_rd_atom=" << devAttr.orig_attr.max_qp_init_rd_atom
     << " caps.max_qp_wr=" << devAttr.orig_attr.max_qp_wr
     << " caps.max_sge=" << devAttr.orig_attr.max_sge
     << " port.active_mtu=" << static_cast<int>(portAttr.active_mtu)
     << " port.gid_tbl_len=" << static_cast<int>(portAttr.gid_tbl_len)
     << " port.link_layer=" << LinkLayerName(portAttr.link_layer) << '('
     << static_cast<int>(portAttr.link_layer) << ')';
  return os.str();
}

}  // namespace

/* ---------------------------------------------------------------------------------------------- */
/*                                      IBVerbsDeviceContext                                      */
/* ---------------------------------------------------------------------------------------------- */
IBVerbsDeviceContext::IBVerbsDeviceContext(RdmaDevice* rdma_device, ibv_pd* inPd)
    : RdmaDeviceContext(rdma_device, inPd) {}

IBVerbsDeviceContext::~IBVerbsDeviceContext() {
  std::lock_guard<std::mutex> lock(poolMu);
  for (auto& it : gpuDirectResources) {
    if (it.second.atomicIbufMr != nullptr) ibv_dereg_mr(it.second.atomicIbufMr);
    if (it.second.atomicIbufAddr != nullptr) (void)hipFree(it.second.atomicIbufAddr);
    for (auto mapping = it.second.mappings.rbegin(); mapping != it.second.mappings.rend();
         ++mapping) {
      if (mapping->hostBase != nullptr) (void)hipHostUnregister(mapping->hostBase);
    }
  }
  gpuDirectResources.clear();
  for (auto& it : qpPool) ibv_destroy_qp(it.second);
  for (auto& it : cqPool) ibv_destroy_cq(it.second);
  for (auto* compCh : compChPool) {
    if (compCh) ibv_destroy_comp_channel(compCh);
  }
}

void IBVerbsDeviceContext::PopulateMlx5GpuDirectHandles(
    RdmaEndpoint& endpoint, const RdmaEndpointConfig& config) {
  if (!UseIbverbsGpuDirect() || !config.onGpu ||
      endpoint.vendorId != RdmaDeviceVendorId::Mellanox) {
    return;
  }

  mlx5dv_qp directQp{};
  mlx5dv_cq directCq{};
  mlx5dv_obj directObject{};
  directObject.qp.in = endpoint.ibvHandle.qp;
  directObject.qp.out = &directQp;
  directObject.cq.in = endpoint.ibvHandle.cq;
  directObject.cq.out = &directCq;
  if (mlx5dv_init_obj(&directObject, MLX5DV_OBJ_QP | MLX5DV_OBJ_CQ) != 0) {
    throw std::runtime_error("mlx5dv_init_obj(QP|CQ) failed for ibverbs GPU-direct endpoint");
  }
  auto IsPowerOfTwo = [](uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
  };
  if (directQp.sq.stride != 64 || !IsPowerOfTwo(directQp.sq.wqe_cnt)) {
    throw std::runtime_error(
        "ibverbs GPU-direct requires a power-of-two mlx5 SQ with 64-byte WQEs");
  }
  if (directCq.cqe_size != 64 || !IsPowerOfTwo(directCq.cqe_cnt)) {
    throw std::runtime_error(
        "ibverbs GPU-direct requires a power-of-two mlx5 CQ with 64-byte CQEs");
  }

  struct RequestedRange {
    uintptr_t begin;
    uintptr_t end;
  };
  constexpr uintptr_t kPageSize = 4096;
  auto PageRange = [&](void* pointer, size_t length) -> RequestedRange {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    return {address & ~(kPageSize - 1),
            (address + length + kPageSize - 1) & ~(kPageSize - 1)};
  };

  std::vector<RequestedRange> requested;
  auto AddRange = [&](void* pointer, size_t length) {
    if (pointer != nullptr && length != 0) requested.push_back(PageRange(pointer, length));
  };
  AddRange(directQp.sq.buf,
           static_cast<size_t>(directQp.sq.wqe_cnt) * directQp.sq.stride);
  AddRange(directQp.rq.buf,
           static_cast<size_t>(directQp.rq.wqe_cnt) * directQp.rq.stride);
  AddRange(directQp.dbrec, 8);
  AddRange(directQp.bf.reg, directQp.bf.size);
  AddRange(directCq.buf,
           static_cast<size_t>(directCq.cqe_cnt) * directCq.cqe_size);
  AddRange(directCq.dbrec, 8);
  std::sort(requested.begin(), requested.end(),
            [](const RequestedRange& left, const RequestedRange& right) {
              return std::tie(left.begin, left.end) < std::tie(right.begin, right.end);
            });

  std::vector<RequestedRange> merged;
  for (const RequestedRange& range : requested) {
    if (merged.empty() || range.begin > merged.back().end) {
      merged.push_back(range);
    } else {
      merged.back().end = std::max(merged.back().end, range.end);
    }
  }

  GpuDirectResources resources;
  struct DeviceMapping {
    uintptr_t hostBegin;
    uintptr_t hostEnd;
    uintptr_t deviceBegin;
  };
  std::vector<DeviceMapping> deviceMappings;
  try {
    for (const RequestedRange& range : merged) {
      void* hostBase = reinterpret_cast<void*>(range.begin);
      const size_t length = range.end - range.begin;
      HIP_RUNTIME_CHECK(hipHostRegister(
          hostBase, length, hipHostRegisterPortable | hipHostRegisterMapped));
      void* deviceBase = nullptr;
      HIP_RUNTIME_CHECK(hipHostGetDevicePointer(&deviceBase, hostBase, 0));
      resources.mappings.push_back({hostBase, length});
      deviceMappings.push_back(
          {range.begin, range.end, reinterpret_cast<uintptr_t>(deviceBase)});
    }

    auto DeviceAlias = [&](void* hostPointer) -> void* {
      const uintptr_t address = reinterpret_cast<uintptr_t>(hostPointer);
      for (const DeviceMapping& mapping : deviceMappings) {
        if (address >= mapping.hostBegin && address < mapping.hostEnd) {
          return reinterpret_cast<void*>(mapping.deviceBegin + address - mapping.hostBegin);
        }
      }
      throw std::runtime_error("ibverbs GPU-direct host pointer is outside registered mappings");
    };

    endpoint.wqHandle.sqAddr = DeviceAlias(directQp.sq.buf);
    endpoint.wqHandle.rqAddr = DeviceAlias(directQp.rq.buf);
    endpoint.wqHandle.dbrRecAddr = DeviceAlias(directQp.dbrec);
    endpoint.wqHandle.dbrAddr = DeviceAlias(directQp.bf.reg);
    endpoint.wqHandle.rqdbrAddr = DeviceAlias(directQp.dbrec);
    endpoint.wqHandle.sqWqeNum = directQp.sq.wqe_cnt;
    endpoint.wqHandle.rqWqeNum = directQp.rq.wqe_cnt;

    endpoint.cqHandle.cqAddr = DeviceAlias(directCq.buf);
    endpoint.cqHandle.dbrRecAddr = DeviceAlias(directCq.dbrec);
    endpoint.cqHandle.consIdx = 0;
    endpoint.cqHandle.needConsIdx = 0;
    endpoint.cqHandle.cqeNum = directCq.cqe_cnt;
    endpoint.cqHandle.cqeSize = directCq.cqe_size;
    endpoint.cqHandle.isCollapsed = 0;

    const size_t atomicIbufSize =
        (RoundUpPowOfTwo(config.atomicIbufSlots) + 1) * ATOMIC_IBUF_SLOT_SIZE;
    HIP_RUNTIME_CHECK(hipExtMallocWithFlags(&resources.atomicIbufAddr, atomicIbufSize,
                                            hipDeviceMallocUncached));
    HIP_RUNTIME_CHECK(hipMemset(resources.atomicIbufAddr, 0, atomicIbufSize));
    const int atomicAccess = MaybeAddRelaxedOrderingFlag(
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
        IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC);
    resources.atomicIbufMr =
        ibv_reg_mr(pd, resources.atomicIbufAddr, atomicIbufSize, atomicAccess);
    if (resources.atomicIbufMr == nullptr) {
      throw std::runtime_error("ibv_reg_mr(atomicIbuf) failed for ibverbs GPU-direct endpoint");
    }
    endpoint.atomicIbuf.addr = reinterpret_cast<uintptr_t>(resources.atomicIbufAddr);
    endpoint.atomicIbuf.lkey = resources.atomicIbufMr->lkey;
    endpoint.atomicIbuf.rkey = resources.atomicIbufMr->rkey;
    endpoint.atomicIbuf.nslots = RoundUpPowOfTwo(config.atomicIbufSlots);

    MORI_APP_INFO(
        "ibverbs GPU-direct endpoint qpn={} sq={}x{} rq={}x{} cq={}x{} bf={} dbrec={}",
        endpoint.handle.qpn, directQp.sq.wqe_cnt, directQp.sq.stride,
        directQp.rq.wqe_cnt, directQp.rq.stride, directCq.cqe_cnt,
        directCq.cqe_size, directQp.bf.reg,
        static_cast<void*>(directQp.dbrec));
    gpuDirectResources.emplace(endpoint.handle.qpn, std::move(resources));
  } catch (...) {
    if (resources.atomicIbufMr != nullptr) ibv_dereg_mr(resources.atomicIbufMr);
    if (resources.atomicIbufAddr != nullptr) (void)hipFree(resources.atomicIbufAddr);
    for (auto mapping = resources.mappings.rbegin(); mapping != resources.mappings.rend();
         ++mapping) {
      if (mapping->hostBase != nullptr) (void)hipHostUnregister(mapping->hostBase);
    }
    throw;
  }
}

void IBVerbsDeviceContext::DestroyGpuDirectResources(uint32_t qpn) noexcept {
  auto resource = gpuDirectResources.find(qpn);
  if (resource == gpuDirectResources.end()) return;
  if (resource->second.atomicIbufMr != nullptr)
    ibv_dereg_mr(resource->second.atomicIbufMr);
  if (resource->second.atomicIbufAddr != nullptr)
    (void)hipFree(resource->second.atomicIbufAddr);
  for (auto mapping = resource->second.mappings.rbegin();
       mapping != resource->second.mappings.rend(); ++mapping) {
    if (mapping->hostBase != nullptr) (void)hipHostUnregister(mapping->hostBase);
  }
  gpuDirectResources.erase(resource);
}

RdmaEndpoint IBVerbsDeviceContext::CreateRdmaEndpoint(const RdmaEndpointConfig& config) {
  ibv_context* context = GetIbvContext();
  const ibv_device_attr_ex* deviceAttr = GetRdmaDevice()->GetDeviceAttr();
  const ibv_port_attr* portAttr = GetRdmaDevice()->GetPortAttr(config.portId);
  if (deviceAttr == nullptr || portAttr == nullptr || portAttr->max_msg_sz == 0) {
    throw std::runtime_error("RDMA device max_msg_sz is unavailable");
  }

  RdmaEndpoint endpoint;
  endpoint.vendorId = ToRdmaDeviceVendorId(deviceAttr->orig_attr.vendor_id);
  endpoint.maxMsgSize = portAttr->max_msg_sz;
  endpoint.handle.psn = 0;
  endpoint.handle.portId = config.portId;
  endpoint.handle.maxSge = config.maxMsgSge;

  assert(portAttr);
  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    endpoint.handle.ib.lid = portAttr->lid;
  } else if (portAttr->link_layer == IBV_LINK_LAYER_ETHERNET) {
    GidSelectionResult gidSelection =
        AutoSelectGidIndex(context, config.portId, portAttr, config.gidIdx);
    if (gidSelection.gidIdx < 0 || !gidSelection.valid) {
      throw std::runtime_error("failed to select a valid RDMA GID index");
    }

    memcpy(endpoint.handle.eth.gid, gidSelection.gid.raw, 16);
    endpoint.handle.eth.gidIdx = gidSelection.gidIdx;
  } else {
    throw std::runtime_error("unsupported RDMA link layer " +
                             std::to_string(static_cast<int>(portAttr->link_layer)));
  }

  // TODO: we need to add more options in config, include min cqe num for ib_create_cq
  endpoint.ibvHandle.compCh = config.withCompChannel ? ibv_create_comp_channel(context) : nullptr;
  if (config.withCompChannel && !endpoint.ibvHandle.compCh) {
    // Capture errno immediately: any intervening libc call before we read it
    // (e.g. isatty() inside the logging sink when stderr is not a tty, which
    // sets ENOTTY) would otherwise overwrite the real failure code.
    const int err = errno;
    MORI_APP_ERROR("ibv_create_comp_channel failed: errno={} ({}); dev={}", err, strerror(err),
                   GetRdmaDevice()->Name());
    throw std::runtime_error("ibv_create_comp_channel failed: " + std::string(strerror(err)));
  }

  endpoint.ibvHandle.cq =
      ibv_create_cq(context, config.maxCqeNum, NULL, endpoint.ibvHandle.compCh, 0);
  if (!endpoint.ibvHandle.cq) {
    // Capture errno before the lock/log path can overwrite it (see the note at
    // the comp-channel site).
    const int err = errno;
    size_t cqPoolSize = 0;
    {
      std::lock_guard<std::mutex> lock(poolMu);
      cqPoolSize = cqPool.size();
    }
    MORI_APP_ERROR(
        "ibv_create_cq failed: errno={} ({}); dev={} max_cqe={} dev_max_cqe={} cqs_in_pool={}", err,
        strerror(err), GetRdmaDevice()->Name(), config.maxCqeNum, deviceAttr->orig_attr.max_cqe,
        cqPoolSize);
    if (endpoint.ibvHandle.compCh) ibv_destroy_comp_channel(endpoint.ibvHandle.compCh);
    throw std::runtime_error("ibv_create_cq failed: " + std::string(strerror(err)));
  }

  // TODO: should also manage the lifecycle of completion channel && srq
  if (config.withCompChannel)
    assert(endpoint.ibvHandle.compCh &&
           (endpoint.ibvHandle.cq->channel == endpoint.ibvHandle.compCh));

  if (config.maxMsgSge > GetRdmaDevice()->GetDeviceAttr()->orig_attr.max_sge) {
    throw std::runtime_error("requested RDMA maxMsgSge exceeds device max_sge");
  }
  endpoint.ibvHandle.srq = config.enableSrq ? CreateRdmaSrqIfNx(config) : nullptr;

  uint32_t maxRecvWr = config.maxRecvWr != 0 ? config.maxRecvWr : config.maxMsgsNum;
  ibv_qp_init_attr qpAttr = {.send_cq = endpoint.ibvHandle.cq,
                             .recv_cq = endpoint.ibvHandle.cq,
                             .srq = endpoint.ibvHandle.srq,
                             .cap =
                                 {
                                     .max_send_wr = config.maxMsgsNum,
                                     .max_recv_wr = maxRecvWr,
                                     .max_send_sge = config.maxMsgSge,
                                     .max_recv_sge = config.maxMsgSge,
                                     .max_inline_data = config.maxInlineData,
                                 },
                             .qp_type = IBV_QPT_RC};
  endpoint.ibvHandle.qp = ibv_create_qp(pd, &qpAttr);
  if (!endpoint.ibvHandle.qp) {
    // Capture errno immediately. Otherwise the lock/log path below overwrites
    // it before we read it: spdlog's console sink calls isatty(stderr), which
    // sets errno=ENOTTY when stderr is not a terminal, masking the real error
    // (commonly EINVAL when the requested QP capacity — max_send_wr x per-WQE
    // size from sge+inline — exceeds the device's per-QP work-queue budget).
    const int err = errno;
    size_t qpPoolSize = 0;
    {
      std::lock_guard<std::mutex> lock(poolMu);
      qpPoolSize = qpPool.size();
    }
    MORI_APP_ERROR(
        "ibv_create_qp failed: errno={} ({}); dev={} port={} max_send_wr={} max_recv_wr={} "
        "max_send_sge={} max_cqe={} dev_caps(max_qp_wr={} max_qp={} max_cqe={}) qps_in_pool={}",
        err, strerror(err), GetRdmaDevice()->Name(), config.portId, qpAttr.cap.max_send_wr,
        qpAttr.cap.max_recv_wr, qpAttr.cap.max_send_sge, config.maxCqeNum,
        deviceAttr->orig_attr.max_qp_wr, deviceAttr->orig_attr.max_qp,
        deviceAttr->orig_attr.max_cqe, qpPoolSize);
    ibv_destroy_cq(endpoint.ibvHandle.cq);
    if (endpoint.ibvHandle.compCh) ibv_destroy_comp_channel(endpoint.ibvHandle.compCh);
    throw std::runtime_error("ibv_create_qp failed: " + std::string(strerror(err)));
  }
  endpoint.handle.qpn = endpoint.ibvHandle.qp->qp_num;

  try {
    PopulateMlx5GpuDirectHandles(endpoint, config);
  } catch (...) {
    ibv_destroy_qp(endpoint.ibvHandle.qp);
    ibv_destroy_cq(endpoint.ibvHandle.cq);
    if (endpoint.ibvHandle.compCh != nullptr)
      ibv_destroy_comp_channel(endpoint.ibvHandle.compCh);
    throw;
  }

  if (config.enableSrq)
    assert(endpoint.ibvHandle.srq && (endpoint.ibvHandle.qp->srq == endpoint.ibvHandle.srq));

  {
    std::lock_guard<std::mutex> lock(poolMu);
    cqPool.insert({endpoint.ibvHandle.cq, endpoint.ibvHandle.cq});
    qpPool.insert({endpoint.ibvHandle.qp->qp_num, endpoint.ibvHandle.qp});
    if (endpoint.ibvHandle.compCh) {
      compChPool.push_back(endpoint.ibvHandle.compCh);
    }
  }
  return endpoint;
}

void IBVerbsDeviceContext::ConnectEndpoint(const RdmaEndpointHandle& local,
                                           const RdmaEndpointHandle& remote, uint32_t qpId) {
  (void)qpId;
  ibv_qp_attr attr{};
  int flags;

  const ibv_device_attr_ex* devAttr = GetRdmaDevice()->GetDeviceAttr();
  const ibv_port_attr* portAttr = GetRdmaDevice()->GetPortAttr(local.portId);
  if (devAttr == nullptr || portAttr == nullptr) {
    throw RdmaError(EINVAL, "RDMA device or port attributes are unavailable");
  }

  ibv_qp* qp = nullptr;
  {
    std::lock_guard<std::mutex> lock(poolMu);
    auto qpIt = qpPool.find(local.qpn);
    if (qpIt != qpPool.end()) qp = qpIt->second;
  }
  if (qp == nullptr) {
    throw RdmaError(ENOENT, "local RDMA QP is not in the ibverbs pool");
  }

  auto ModifyOrThrow = [&](const char* transition, ibv_qp_attr& qpAttr, int modifyFlags) {
    int ret = ibv_modify_qp(qp, &qpAttr, modifyFlags);
    if (ret == 0) return;

    int err = ret > 0 ? ret : errno;
    if (err == 0) err = EIO;
    std::string detail = DescribeQpTransition(transition, qpAttr, modifyFlags, *devAttr, *portAttr,
                                              local, GetRdmaDevice()->Name().c_str());
    MORI_APP_ERROR("ibv_modify_qp({}) failed: err={} ({}); {}", transition, err, strerror(err),
                   detail);
    throw RdmaError(err, "ibv_modify_qp(" + std::string(transition) +
                             ") failed: " + std::string(strerror(err)));
  };

  // INIT
  attr = {};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = local.portId;
  attr.pkey_index = 0;
  attr.qp_access_flags = MR_DEFAULT_ACCESS_FLAG;
  flags = IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS;
  ModifyOrThrow("INIT", attr, flags);

  // RTR
  attr = {};
  attr.qp_state = IBV_QPS_RTR;
  {
    ibv_mtu path_mtu = portAttr->active_mtu;
    const char* envMtu = std::getenv("MORI_IB_PATH_MTU");
    if (envMtu != nullptr) {
      int mtuBytes = std::atoi(envMtu);
      if (mtuBytes == 256)
        path_mtu = IBV_MTU_256;
      else if (mtuBytes == 512)
        path_mtu = IBV_MTU_512;
      else if (mtuBytes == 1024)
        path_mtu = IBV_MTU_1024;
      else if (mtuBytes == 2048)
        path_mtu = IBV_MTU_2048;
      else if (mtuBytes == 4096)
        path_mtu = IBV_MTU_4096;
      else
        MORI_APP_WARN("Ignore invalid MORI_IB_PATH_MTU={} (allowed: 256/512/1024/2048/4096)",
                      envMtu);
      MORI_APP_INFO("MORI_IB_PATH_MTU override: {} bytes (ibv_mtu={})", mtuBytes, (int)path_mtu);
    }
    attr.path_mtu = path_mtu;
  }
  attr.dest_qp_num = remote.qpn;
  attr.rq_psn = 0;
  attr.max_dest_rd_atomic = devAttr->orig_attr.max_qp_rd_atom;
  attr.min_rnr_timer = 12;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = local.portId;
  std::optional<uint8_t> sl = ReadIoServiceLevelEnv();
  if (!sl.has_value()) {
    sl = ReadRdmaServiceLevelEnv();
  }
  attr.ah_attr.sl = sl.value_or(0);

  bool disableIoTc = ReadIoTrafficClassDisableEnv();
  if (!disableIoTc) {
    std::optional<uint8_t> tc = ReadIoTrafficClassEnv();
    if (!tc.has_value()) {
      tc = ReadRdmaTrafficClassEnv();
    }
    if (tc.has_value()) {
      attr.ah_attr.grh.traffic_class = tc.value();
    }
  }
  MORI_APP_INFO("ibverbs attr.ah_attr.sl:{} attr.ah_attr.grh.traffic_class:{}", attr.ah_attr.sl,
                attr.ah_attr.grh.traffic_class);

  if (portAttr->link_layer == IBV_LINK_LAYER_INFINIBAND) {
    attr.ah_attr.dlid = remote.ib.lid;
  } else if (portAttr->link_layer == IBV_LINK_LAYER_ETHERNET) {
    attr.ah_attr.is_global = 1;
    union ibv_gid dgid;
    memcpy(dgid.raw, remote.eth.gid, 16);
    attr.ah_attr.grh.dgid = dgid;
    attr.ah_attr.grh.sgid_index = local.eth.gidIdx;
    attr.ah_attr.grh.hop_limit = 16;
  }
  flags = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER | IBV_QP_AV;
  ModifyOrThrow("RTR", attr, flags);

  // RTS
  attr = {};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = 0;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = devAttr->orig_attr.max_qp_init_rd_atom;
  flags = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
          IBV_QP_MAX_QP_RD_ATOMIC;
  ModifyOrThrow("RTS", attr, flags);
}

bool IBVerbsDeviceContext::DestroyRdmaEndpointNoThrow(const RdmaEndpoint& ep) noexcept {
  bool ok = true;
  try {
    std::lock_guard<std::mutex> lock(poolMu);

    DestroyGpuDirectResources(ep.handle.qpn);

    auto qpIt = qpPool.find(ep.handle.qpn);
    if (qpIt != qpPool.end() && qpIt->second != nullptr) {
      int ret = ibv_destroy_qp(qpIt->second);
      if (ret == 0) {
        qpPool.erase(qpIt);
      } else {
        int err = ret > 0 ? ret : errno;
        if (err == 0) err = EIO;
        MORI_APP_ERROR("ibv_destroy_qp failed during endpoint rollback: qpn={} err={} ({})",
                       ep.handle.qpn, err, strerror(err));
        ok = false;
      }
    }

    if (ep.ibvHandle.cq != nullptr) {
      auto cqIt = cqPool.find(ep.ibvHandle.cq);
      if (cqIt != cqPool.end()) {
        int ret = ibv_destroy_cq(cqIt->second);
        if (ret == 0) {
          cqPool.erase(cqIt);
        } else {
          int err = ret > 0 ? ret : errno;
          if (err == 0) err = EIO;
          MORI_APP_ERROR("ibv_destroy_cq failed during endpoint rollback: cq={} err={} ({})",
                         static_cast<void*>(ep.ibvHandle.cq), err, strerror(err));
          ok = false;
        }
      }
    }

    if (ep.ibvHandle.compCh != nullptr) {
      auto compIt = std::find(compChPool.begin(), compChPool.end(), ep.ibvHandle.compCh);
      if (compIt != compChPool.end()) {
        int ret = ibv_destroy_comp_channel(ep.ibvHandle.compCh);
        if (ret == 0) {
          compChPool.erase(compIt);
        } else {
          int err = ret > 0 ? ret : errno;
          if (err == 0) err = EIO;
          MORI_APP_ERROR(
              "ibv_destroy_comp_channel failed during endpoint rollback: comp_ch={} err={} ({})",
              static_cast<void*>(ep.ibvHandle.compCh), err, strerror(err));
          ok = false;
        }
      }
    }
  } catch (const std::exception& e) {
    MORI_APP_ERROR("DestroyRdmaEndpointNoThrow caught exception during rollback: {}", e.what());
    return false;
  } catch (...) {
    MORI_APP_ERROR("DestroyRdmaEndpointNoThrow caught unknown exception during rollback");
    return false;
  }
  return ok;
}

/* ---------------------------------------------------------------------------------------------- */
/*                                          IBVerbsDevice                                         */
/* ---------------------------------------------------------------------------------------------- */
IBVerbsDevice::IBVerbsDevice(ibv_device* device) : RdmaDevice(device) {}
IBVerbsDevice::~IBVerbsDevice() {}

RdmaDeviceContext* IBVerbsDevice::CreateRdmaDeviceContext() {
  ibv_pd* pd = ibv_alloc_pd(defaultContext);
  if (!pd) {
    MORI_APP_ERROR("ibv_alloc_pd failed: errno={} ({}); dev={}", errno, strerror(errno), Name());
    throw std::runtime_error("ibv_alloc_pd failed: " + std::string(strerror(errno)));
  }
  return new IBVerbsDeviceContext(this, pd);
}

}  // namespace application
}  // namespace mori

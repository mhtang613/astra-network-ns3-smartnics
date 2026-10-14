/*
 * Congestion-control event trace.
 *
 * When g_cc_trace is non-NULL, the switch and NIC models append one line per
 * congestion-control event to it. Set it from the simulation driver (see
 * scratch/common.h, config key CC_TRACE_FILE). Leave it NULL for no tracing.
 *
 * Line format: "<time_ns> <EVENT> key=value key=value ..."
 *   ECN_MARK  sw= port= pg= src= dst= qlen=            (switch marks a packet)
 *   CNP_SENT  node= src= dst= sport= seq=               (receiver NIC sets CNP on ACK)
 *   CNP_RCVD  node= src= dst= sport= rate_gbps= alpha= target_gbps= first=
 *   RATE_DEC  node= src= dst= sport= old_gbps= new_gbps= alpha= target_gbps=
 *   RATE_INC  node= src= dst= sport= stage= old_gbps= new_gbps= alpha= target_gbps=
 *             (stage = fast_recovery | active | hyper)
 * src/dst are node ids derived from the flow's IP addresses.
 */
#ifndef CC_TRACE_H
#define CC_TRACE_H

#include <cstdio>
#include <cstdint>

namespace ns3 {

extern FILE* g_cc_trace;

// Matches scratch/common.h::node_id_to_ip / ip_to_node_id.
inline uint32_t CcTraceNodeId(uint32_t ip) { return (ip >> 8) & 0xffff; }

} // namespace ns3

#endif /* CC_TRACE_H */

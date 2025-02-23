#include "ttm.h"
#include "shader.h"
#include <float.h>
#include <limits.h>
#include <string.h>
#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_sim.h"
#include "../statwrapper.h"
#include "addrdec.h"
#include "dram.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "icnt_wrapper.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader_trace.h"
#include "stat-tool.h"
#include "traffic_breakdown.h"
#include "visualizer.h"
#include <queue>


#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void scalar_shader_core_ctx::issue_warp(register_set &pipe_reg_set,
                                        const warp_inst_t *next_inst,
                                        const active_mask_t &active_mask,
                                        unsigned warp_id, unsigned sch_id) {    
    fprintf(stdout, "reached scalar issue warp\n");
}

void scalar_shader_core_ctx::create_schedulers() {
    shader_core_ctx::create_schedulers();
    m_fetched_register_board = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu); // Fetched Register board is effectively another scoreboard
}

void scalar_shader_core_ctx::writeback() {
    
}


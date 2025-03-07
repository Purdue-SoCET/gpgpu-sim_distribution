#include "ttm.h"
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
#include <vector>


#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
    
void divergent_tid_table::reset_entry(unsigned warp_id) {
    divergent_tid_table_arr[warp_id].valid = false; 
    divergent_tid_table_arr[warp_id].scalar_tid = -1; 
    divergent_tid_table_arr[warp_id].simt_tid = -1; 
    divergent_tid_table_arr[warp_id].simt_wid = -1; 
}

void divergent_tid_table::set_entry(unsigned scalar_tid, unsigned simt_tid, unsigned simt_wid) {
    divergent_tid_table_arr[scalar_tid].valid = true; 
    divergent_tid_table_arr[scalar_tid].scalar_tid = scalar_tid; 
    divergent_tid_table_arr[scalar_tid].simt_tid = simt_tid; 
    divergent_tid_table_arr[scalar_tid].simt_wid = simt_wid;  
}

div_tid_table_entry divergent_tid_table::get_entry(unsigned scalar_tid) {
    return divergent_tid_table_arr[scalar_tid]; 
}

int divergent_tid_table::find_free_entry() {
    for (int i = 0; i < num_entries; i++) {
        if (!divergent_tid_table_arr[i].valid) { // Nothing in that spot
            return i; 
        }

    }
    return -1; 
}

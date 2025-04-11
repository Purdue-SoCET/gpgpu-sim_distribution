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
#include "../abstract_hardware_model.h"


#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
    
void divergent_tid_table::reset_entry(unsigned warp_id) {
    divergent_tid_table_arr[warp_id].valid = false; 
    divergent_tid_table_arr[warp_id].reconverge = false; 
    divergent_tid_table_arr[warp_id].reconverge_done = false; 
    divergent_tid_table_arr[warp_id].scalar_tid = -1; 
    divergent_tid_table_arr[warp_id].simt_tid = -1; 
    divergent_tid_table_arr[warp_id].simt_wid = -1; 
    divergent_tid_table_arr[warp_id].simt_rpc = -1; 
}

void divergent_tid_table::invalidate_entry(unsigned warp_id) {
    divergent_tid_table_arr[warp_id].valid = false; 
}

void divergent_tid_table::print() {
    for (int warp_id = 0; warp_id < SCALAR_BANDWIDTH; warp_id++) {
        printf("valid=%d, reconverge=%d, reconverge_done=%d, scalar_tid=%d, simt_tid=%d, simt_wid=%d, simt_rpc=%x\n", 
        divergent_tid_table_arr[warp_id].valid, divergent_tid_table_arr[warp_id].reconverge, divergent_tid_table_arr[warp_id].reconverge_done,
        divergent_tid_table_arr[warp_id].scalar_tid, divergent_tid_table_arr[warp_id].simt_tid, divergent_tid_table_arr[warp_id].simt_wid,
        divergent_tid_table_arr[warp_id].simt_rpc); 
    }
    printf("\n"); 
}

void divergent_tid_table::set_entry(unsigned scalar_tid, unsigned simt_tid, unsigned simt_wid, address_type simt_rpc) {
    divergent_tid_table_arr[scalar_tid].valid = true; 
    divergent_tid_table_arr[scalar_tid].reconverge = false; 
    divergent_tid_table_arr[scalar_tid].reconverge_done = false; 
    divergent_tid_table_arr[scalar_tid].scalar_tid = scalar_tid; 
    divergent_tid_table_arr[scalar_tid].simt_tid = simt_tid; 
    divergent_tid_table_arr[scalar_tid].simt_wid = simt_wid;  
    divergent_tid_table_arr[scalar_tid].simt_rpc = simt_rpc; 
}

div_tid_table_entry divergent_tid_table::get_entry(unsigned scalar_tid) {
    return divergent_tid_table_arr[scalar_tid]; 
}

void divergent_tid_table::set_reconverge(unsigned scalar_tid, bool reconverge) {
    divergent_tid_table_arr[scalar_tid].reconverge = reconverge;   
}

void divergent_tid_table::set_reconverge_done(unsigned scalar_tid, bool reconverge_done) {
    divergent_tid_table_arr[scalar_tid].reconverge_done = reconverge_done;   
}

int divergent_tid_table::find_free_entry() {
    for (int i = 0; i < num_entries; i++) {
        if (!divergent_tid_table_arr[i].valid) { // Nothing in that spot
            return i; 
        }

    }
    return -1; 
}

// Returns the thread associated with warp if it is in table 
unsigned divergent_tid_table::is_wid_in_table(unsigned simt_wid) {
    for (int i = 0; i < num_entries; i++) {
        if (divergent_tid_table_arr[i].simt_wid == simt_wid && divergent_tid_table_arr[i].valid) { 
            return divergent_tid_table_arr[i].simt_tid; 
        }

    }
    return -1; 
} 

unsigned divergent_tid_table::is_wid_ready_to_reconv(unsigned simt_wid) {
    for (int i = 0; i < num_entries; i++) {
        if (divergent_tid_table_arr[i].simt_wid == simt_wid && !divergent_tid_table_arr[i].valid && divergent_tid_table_arr[i].reconverge && divergent_tid_table_arr[i].reconverge_done) { 
            return divergent_tid_table_arr[i].simt_tid; 
        }
    }
    return -1; 
}

unsigned divergent_tid_table::simt_to_scalar(unsigned simt_wid, unsigned simt_tid) {
    for (int i = 0; i < num_entries; i++) {
        if (divergent_tid_table_arr[i].simt_wid == simt_wid && divergent_tid_table_arr[i].simt_tid == simt_tid) { 
            return i; 
        }
    }
    return -1; 
}
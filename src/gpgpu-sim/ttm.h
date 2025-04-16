#ifndef TTM_H
#define TTM_H

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>

//#include "../cuda-sim/ptx.tab.h"

#include "../abstract_hardware_model.h"
// #include "shader.h"
// #include "delayqueue.h"
// #include "dram.h"
// #include "gpu-cache.h"
// #include "mem_fetch.h"
#include "scoreboard.h"
// #include "stack.h"
// #include "stats.h"
// #include "traffic_breakdown.h"
#include <queue>
#include <vector>

typedef struct div_tid_table_entry {
    bool valid;  
    unsigned scalar_tid; 
    unsigned simt_tid;
    unsigned simt_wid; 
    address_type simt_rpc;
} div_tid_table_entry;

typedef struct ready_table_entry {
    bool ready;
    unsigned tid; 
} ready_table_entry; 

enum return_fsm_states {
    IDLE,
    PULL,
    READING,
    LOOKUP,
    SEND
};

typedef struct wrb_entry {
    unsigned wid;
    unsigned regnum; 
} wrb_entry;


class ready_table {
    // Rows are SIMT core warp ID
    // Each warp ID has a ready bit and the thread ID that needs to be reasserted 
    private:
    std::vector<ready_table_entry> ready_table_arr;
    int num_entries; 

    public:
    // Default constructor - starts empty
    ready_table() : num_entries(0) {}

    // Constructor that sets size immediately
    ready_table(int n) {
        resize(n); 
    }

    void resize(int n) {
        num_entries = n; 
        ready_table_arr.resize(n); 
        for (int i = 0; i < n; i++) {
            ready_table_arr[i].ready = false; 
            ready_table_arr[i].tid = -1; 
        }
    } 

    void set_entry(unsigned wid, unsigned tid);
    void reset_entry(unsigned wid); 
    ready_table_entry get_entry(unsigned wid); 
};

class divergent_tid_table {
    private:
    std::vector<div_tid_table_entry> divergent_tid_table_arr;
    int num_entries; 

    public:
    // Default constructor - starts empty
    divergent_tid_table() : num_entries(0) {}

    // Constructor that sets size immediately
    divergent_tid_table(int n) {
        resize(n); 
    }

    void resize(int n) {
        num_entries = n; 
        divergent_tid_table_arr.resize(n); 
        for (int i = 0; i < n; i++) {
            divergent_tid_table_arr[i].valid = false; 
            divergent_tid_table_arr[i].scalar_tid = -1; 
            divergent_tid_table_arr[i].simt_tid = -1; 
            divergent_tid_table_arr[i].simt_wid = -1; 
            divergent_tid_table_arr[i].simt_rpc = -1; 
        }
    } 
    void reset_entry(unsigned warp_id);
    void set_entry(unsigned scalar_tid, unsigned simt_tid, unsigned simt_wid, address_type simt_rpc); 
    div_tid_table_entry get_entry(unsigned scalar_tid);
    int find_free_entry(); 
    void set_reconverge(unsigned scalar_tid, bool reconverge); 
    void set_reconverge_done(unsigned scalar_tid, bool reconverge_done); 
    void print(); 
    unsigned is_wid_in_table(unsigned simt_wid); 
    void invalidate_entry(unsigned warp_id);
    unsigned simt_to_scalar(unsigned simt_wid, unsigned simt_tid);

};


#endif /* TTM_H */

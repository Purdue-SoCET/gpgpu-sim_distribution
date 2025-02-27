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
#include "shader.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "mem_fetch.h"
#include "scoreboard.h"
#include "stack.h"
#include "stats.h"
#include "traffic_breakdown.h"
#include <queue>
#include <vector>

typedef struct div_tid_table_entry {
    unsigned scalar_tid; 
    unsigned simt_tid;
    unsigned simt_wid; 
} div_tid_table_entry;

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

class divergent_tid_table {
    public:
    std::vector<div_tid_table_entry> divergent_tid_table_arr;

    divergent_tid_table(int n) : divergent_tid_table_arr(n) {} // Instantiate this when cores getting instantiated based on number of warps in scalar core
};

class scalar_shader_core_ctx : public shader_core_ctx {
    public:
        bool steal;
        bool reconverge;
        return_fsm_states curr_state;
        return_fsm_states next_state; 
        wrb_entry curr_wrb_entry; 
        
    // creator:
    scalar_shader_core_ctx(class gpgpu_sim *gpu, class simt_core_cluster *cluster,
        unsigned shader_id, unsigned tpc_id,
        const shader_core_config *config,
        const memory_config *mem_config, shader_core_stats *stats);


    protected:
        virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
            const active_mask_t &active_mask, unsigned warp_id,
            unsigned sch_id) override;

        virtual void create_schedulers() override;

        virtual void writeback() override; 

        void return_fsm_cycle(); 

        void update_return_fsm(); 

        Scoreboard *m_fetched_register_board;
        std::queue<wrb_entry> m_written_register_board; 
};


#endif /* TTM_H */

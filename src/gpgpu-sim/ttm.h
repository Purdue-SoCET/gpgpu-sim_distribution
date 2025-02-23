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

class wrb_entry {
    public:
    unsigned wid;
    unsigned regnum; 
    wrb_entry(unsigned wid, unsigned regnum) : wid(wid), regnum(regnum) {}
};

class scalar_shader_core_ctx : public shader_core_ctx {
    public:
        bool steal;
        bool reconverge;

    protected:
        virtual void issue_warp(register_set &warp, const warp_inst_t *pI,
            const active_mask_t &active_mask, unsigned warp_id,
            unsigned sch_id) override;

        virtual void create_schedulers() override;

        virtual void writeback() override; 

        Scoreboard *m_fetched_register_board;
        std::queue<wrb_entry> m_written_register_board; 
};



#endif /* TTM_H */

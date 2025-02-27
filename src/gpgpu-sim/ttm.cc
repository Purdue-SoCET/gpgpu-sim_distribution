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
#include <vector>


#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// scalar_shader_core_ctx::scalar_shader_core_ctx(class gpgpu_sim *gpu,
//     class simt_core_cluster *cluster,
//     unsigned shader_id, unsigned tpc_id,
//     const shader_core_config *config,
//     const memory_config *mem_config,
//     shader_core_stats *stats)
// : shader_core_ctx(gpu, cluster, shader_id, tpc_id, shader_core_config, memory_config, stats) {

// }

void scalar_shader_core_ctx::issue_warp(register_set &pipe_reg_set,
                                        const warp_inst_t *next_inst,
                                        const active_mask_t &active_mask,
                                        unsigned warp_id, unsigned sch_id) { 
    // Modified instr has src registers become dest registers so checkCollision will flag properly
    warp_inst_t frb_entry = *next_inst; 
    for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
        frb_entry.arch_reg.dst[i] = next_inst->arch_reg.src[i];
    }

    if (m_fetched_register_board->checkCollision(warp_id, &frb_entry)) { // Check if src reg (moved to dst reg) is in fetched register board
        // Set steal signal, add to FRB
        steal = true; 
        m_fetched_register_board->reserveRegisters(&frb_entry); 
    } else { // Otherwise proceed as normal
        shader_core_ctx::issue_warp(pipe_reg_set, next_inst, active_mask, warp_id, sch_id); 
    }
}

void scalar_shader_core_ctx::create_schedulers() {
    shader_core_ctx::create_schedulers();
    m_fetched_register_board = new Scoreboard(m_sid, m_config->max_warps_per_shader, m_gpu); // Fetched Register board is effectively another scoreboard
}

void scalar_shader_core_ctx::writeback() {
  unsigned max_committed_thread_instructions =
      m_config->warp_size *
      (m_config->pipe_widths[EX_WB]);  // from the functional units
  m_stats->m_pipeline_duty_cycle[m_sid] =
      ((float)(m_stats->m_num_sim_insn[m_sid] -
               m_stats->m_last_num_sim_insn[m_sid])) /
      max_committed_thread_instructions;

  m_stats->m_last_num_sim_insn[m_sid] = m_stats->m_num_sim_insn[m_sid];
  m_stats->m_last_num_sim_winsn[m_sid] = m_stats->m_num_sim_winsn[m_sid];

  warp_inst_t **preg = m_pipeline_reg[EX_WB].get_ready();
  warp_inst_t *pipe_reg = (preg == NULL) ? NULL : *preg;
  while (preg and !pipe_reg->empty()) {
    /*
     * Right now, the writeback stage drains all waiting instructions
     * assuming there are enough ports in the register file or the
     * conflicts are resolved at issue.
     */
    /*
     * The operand collector writeback can generally generate a stall
     * However, here, the pipelines should be un-stallable. This is
     * guaranteed because this is the first time the writeback function
     * is called after the operand collector's step function, which
     * resets the allocations. There is one case which could result in
     * the writeback function returning false (stall), which is when
     * an instruction tries to modify two registers (GPR and predicate)
     * To handle this case, we ignore the return value (thus allowing
     * no stalling).
     */

    m_operand_collector.writeback(*pipe_reg);
    unsigned warp_id = pipe_reg->warp_id();
    m_scoreboard->releaseRegisters(pipe_reg);
    m_warp[warp_id]->dec_inst_in_pipeline();

    std::list<unsigned> regs = get_regs_written(*pipe_reg);
    for (unsigned reg : regs) {
        wrb_entry entry;
        entry.wid = warp_id; 
        entry.regnum = reg; 
        m_written_register_board.emplace(entry); 
    }

    warp_inst_complete(*pipe_reg);
    m_gpu->gpu_sim_insn_last_update_sid = m_sid;
    m_gpu->gpu_sim_insn_last_update = m_gpu->gpu_sim_cycle;
    m_last_inst_gpu_sim_cycle = m_gpu->gpu_sim_cycle;
    m_last_inst_gpu_tot_sim_cycle = m_gpu->gpu_tot_sim_cycle;
    pipe_reg->clear();
    preg = m_pipeline_reg[EX_WB].get_ready();
    pipe_reg = (preg == NULL) ? NULL : *preg;
  }
}

void scalar_shader_core_ctx::return_fsm_cycle() {
    curr_state = next_state; 
}

void scalar_shader_core_ctx::update_return_fsm() {
    next_state = curr_state; 
    // switch(curr_state) {
    //     case IDLE:
    //         if (reconverge) {
    //             next_state = PULL; 
    //         }
    //     case PULL:
    //         if (m_written_register_board.empty()) {
    //             next_state = IDLE; 
    //         } else {
    //             curr_wrb_entry = m_written_register_board.front(); 
    //             m_written_register_board.pop();
    //             next_state = READING;
    //         }
    //     case READING:
            
    //         // Use wid and regnum to read from register file
    //     case LOOKUP:
    //         // Reference divergent thread ID table to find where this belongs in SIMT core
    //     case SEND:
    //         // set values to send to SIMT core
    // }
}


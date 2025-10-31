// Shresth Mathur
// Akshath Raghav Ravikiran

// Function definitions for carrying out divergence detection and scalarization heuristic

#include "shader.h"

bool shader_core_ctx::push_scalar_que(scalar_que_entry entry) {
  // scalar_que_entry entry;
  // entry.m_tid = tid;
  // entry.m_warp_id = warp_id;
  // entry.start_pc = start_pc;
  // entry.reconv_pc = reconv_pc;

  if (scalar_que.size() == SCALAR_CORE_CAPACITY) {
    return false; // Failed to push because que is full
  }
  else {
    scalar_que.push_back(entry);
    return true;
  }
}

scalar_que_entry shader_core_ctx::pop_scalar_que() {
  scalar_que_entry top_of_que = scalar_que.front();
  scalar_que.pop_front();
  return top_of_que;
}

void shader_core_ctx::display_scalar_que() {
  fprintf(stdout,"Scalar Que State\n");
  fprintf(stdout,"Warp ID | Thread ID | Start PC | Reconvergence PC\n");
  for(int i=get_scalar_que_ocp()-1; i>=0; i--){
    scalar_que_entry que_entry = scalar_que[i];
    fprintf(stdout,"%d      | %d        | %x       | %x\n",que_entry.m_warp_id,que_entry.m_tid,que_entry.start_pc,que_entry.reconv_pc);
  }
}

void shd_warp_t::get_pcs(unsigned *rpc, unsigned *pc) {
    unsigned tid = m_warp_id * m_warp_size;
    m_shader->get_pdom_stack_top_info(tid, pc, rpc);
}

bool shd_warp_t::in_div_region() {
  unsigned pc, rpc;
  get_pcs(&rpc, &pc);
  return rpc != -1;
}

unsigned shd_warp_t::count_active_threads(active_mask_t thread_mask) {
  unsigned cnt = 0;
  for(int i=0; i<m_warp_size; i++){
    if(thread_mask[i]){
      cnt++;
    }
  }
  return cnt;
}

void shd_warp_t::increment_sat_counters(active_mask_t result_thread_mask) {
  // if we're not in a divergent region, don't update counters
  if (!in_div_region()) return;

  for (int i = 0; i < (int)m_warp_size; i++) {
    if (result_thread_mask[i]) {
      if (sat_counters[i] < SAT_LIMIT) {
        sat_counters[i]++;
      }
    }
  }
}

std::vector<unsigned> shd_warp_t::check_sat_counters() {
  std::vector<unsigned> scalar_tids;

  if (!in_div_region()) return scalar_tids;
  if (sat_counters.empty()) return scalar_tids;

  // collect candidates that have hit saturation and are part of the result mask (and are not already scalarized)
  for (unsigned i = 0; i < m_warp_size; i++) {
    if (sat_counters[i] >= SAT_LIMIT && !scalar_mask[i]) {
      scalar_tids.push_back(i);
      fprintf(stdout, "Thread %u in warp %u has saturated counter %u --> ready for scalarization\n", i, m_warp_id, sat_counters[i]);
      if (scalar_tids.size() >= SCALAR_BANDWIDTH) break; // cap at bandwidth
    }
  }

  return scalar_tids;
}

void shd_warp_t::set_scalar_regs(std::vector<unsigned> scalar_tids) {
  unsigned rpc, pc;
  get_pcs(&rpc, &pc);
  
  for(int i = 0; i < scalar_regs.size(); i++){
    scalar_reg reg = scalar_regs[i];

    if(!(reg.dirty)) { // If the register is not already occupied
      if (scalar_tids.size() == 0) break; // No more threads to scalarize -- THIS LINE MUST STAY HERE DO NOT MOVE IT!!!!
      
      unsigned tid = scalar_tids.back();  // This infers that scalar registers are assigned in LIFO order from the list of candidate thread IDs
      scalar_tids.pop_back();

      sat_counters[tid] = 0; // Reset saturating counter for thread to be scalarized
      scalar_mask[tid] = 1; // Set scalar mask bit after the thread context has been registered
      
      // Setting scalar register params
      reg.m_tid = tid;
      reg.start_pc = pc;
      reg.reconv_pc = rpc;
      reg.dirty = true;

      scalar_regs[i] = reg;

      num_scalarizations++;
      // fprintf(stdout,"Register %d on warp %d has dirty bit %d\n",i,m_warp_id,reg.dirty);
    }
  }
}

void shd_warp_t::cycle_through_scalar_regs() {
  scalar_reg reg = scalar_regs[reg_cntr];
  fprintf(stdout,"Cycling through scalar register %d on warp %d\n",reg_cntr,m_warp_id);
  // fprintf(stdout,"Reg counter value %d\n",reg_cntr);
  // fprintf(stdout,"Scalar Register State for Warp %d\n",m_warp_id);
  // fprintf(stdout,"Thread ID | Start PC | Reconvergence PC | Dirty\n");

  // for(int i=SCALAR_BANDWIDTH-1; i>=0; i--){
  //   scalar_reg que_entry = scalar_regs[i];
  //   fprintf(stdout,"%d        | %x       | %x               | %d\n",que_entry.m_tid,que_entry.start_pc,que_entry.reconv_pc,que_entry.dirty);
  // }

  if(!get_elected_status()) { // If thread has not yet been elected for scalarization, elect it in the following block --> used by RR scheduler (rr_top_level_scheduler)
    if(reg.dirty) { // If register is occupied (has valid thread context)

      // Setting up scalar que entry
      scalar_que_entry entry;
      entry.m_tid = reg.m_tid;
      entry.m_warp_id = m_warp_id;
      entry.start_pc = reg.start_pc;
      entry.reconv_pc = reg.reconv_pc;

      set_elected_status(true); // Now the RR scheduler knows that this thread is ready to be scalarized
      set_elected_thread(entry);
      fprintf(stdout,"Elected thread %d in warp %d to be scalarized\n",reg.m_tid,m_warp_id);

      reg.dirty = 0;  // Mark register as free after pushing to scalar que
      scalar_regs[reg_cntr] = reg;
      // m_shader->display_scalar_que();
    }
    if (reg_cntr == SCALAR_BANDWIDTH-1) {
      reg_cntr = 0;
    }
    else {
      reg_cntr++;
    }
  }
}

void shader_core_ctx::rr_top_level_scheduler() {
  shd_warp_t * warp = m_warp[warp_cntr];
  fprintf(stdout,"RR Top Level scheduler checking Warp %d\n",warp_cntr);
  // fprintf(stdout,"Top Level scheduler checking Warp %d\n",warp_cntr);
  if (warp->get_elected_status()) {
    bool pushed = push_scalar_que(warp->get_elected_thread());
    
    if (pushed) {
      fprintf(stdout,"Scalarized thread %d in warp %d\n",warp->get_elected_thread().m_tid, warp->get_elected_thread().m_warp_id);
      warp->set_elected_status(false);
      display_scalar_que();
    }
  }

  // if only 1 warp, no need to cycle -- if multiple warps, cycle through them by incrementing warp counter to maintain RR fairness
  if (warp_cntr == m_config->max_warps_per_shader-1) {
    warp_cntr = 0;
  }
  else {
    warp_cntr++;
  }
}
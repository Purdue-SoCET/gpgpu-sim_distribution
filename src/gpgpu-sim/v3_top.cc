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

  if (scalar_que->size() >= SCALAR_CORE_CAPACITY) {
    return false; // Failed to push because que is full
  }
  else {
    scalar_que->push_back(entry);
    return true;
  }
}

scalar_que_entry shader_core_ctx::pop_scalar_que() {
  scalar_que_entry top_of_que = scalar_que->front();
  scalar_que->pop_front();
  return top_of_que;
}

void shader_core_ctx::display_scalar_que() {
  // fprintf(stdout,"Scalar Que State\n");
  // fprintf(stdout,"Warp ID | Thread ID | Start PC | Reconvergence PC\n");
  for(int i=get_scalar_que_ocp()-1; i>=0; i--){
    scalar_que_entry que_entry = (*scalar_que)[i];
    // fprintf(stdout,"%d      | %d        | %x       | %x\n",que_entry.m_warp_id,que_entry.m_tid,que_entry.start_pc,que_entry.reconv_pc);
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
      // fprintf(stdout, "Thread %u in warp %u has saturated counter %u --> ready for scalarization\n", i, m_warp_id, sat_counters[i]);
      if (scalar_tids.size() >= SCALAR_BANDWIDTH) break; // cap at bandwidth
    }
  }

  return scalar_tids;
}

void shd_warp_t::clear_counters(unsigned tid) {
  // Reset all the counting logic so it's ready to push stuff again
  for (unsigned i = 0; i < m_warp_size; i++) {
    sat_counters[i] = 0; 
  }
  scalar_mask[tid] = 0;
  scalar_regs[tid].dirty = 0; 
  scalar_regs[tid].m_tid = 0;
  scalar_regs[tid].start_pc = 0;
  scalar_regs[tid].reconv_pc = 0;
}


unsigned shd_warp_t::set_scalar_regs(std::vector<unsigned> scalar_tids){
  unsigned num_scalarized = 0;
  for(int i = 0; i < scalar_regs.size(); i++){
    scalar_reg reg = scalar_regs[i];
    unsigned rpc, pc;
    get_pcs(&rpc, &pc);

    if(!(reg.dirty)){ // If the register is not already occupied
      if(scalar_tids.size() != 0){
        unsigned tid = scalar_tids.back();
        scalar_tids.pop_back();

        sat_counters[tid] = 0; // Reset Saturating Counters
        scalar_mask[tid] = 1; // Set scalar mask bit after the thread context has been registered
        
        // printf("push to reg when pc is %x\n", pc); 

        reg.m_tid = tid;
        reg.start_pc = pc;
        reg.reconv_pc = rpc;
        reg.dirty = 1;

        scalar_regs[i] = reg;

        num_scalarizations += 1;
        num_scalarized++;

        // fprintf(stdout,"Register %d on warp %d has dirty bit %d\n",i,m_warp_id,reg.dirty);
      }
    }
  }

  return num_scalarized;
}

void shd_warp_t::cycle_through_scalar_regs(){
  scalar_reg reg = scalar_regs[reg_cntr];

  if (!get_elected_status()){ // Waits until elected thread is pushed to scalar que
    if(reg.dirty){
      
      scalar_que_entry entry;
      entry.m_tid = reg.m_tid;
      entry.m_warp_id = m_warp_id;
      entry.start_pc = reg.start_pc;
      entry.reconv_pc = reg.reconv_pc;

      set_elected_status((bool) 1);
      set_elected_thread(entry);

      fprintf(stdout,"Elected to scalarize thread %d in warp %d\n",reg.m_tid,m_warp_id);

      reg.dirty = 0;
      scalar_regs[reg_cntr] = reg;
    }

    if(reg_cntr == SCALAR_BANDWIDTH-1){
      reg_cntr = 0;
    }

    else{
      reg_cntr++;
    }
  }
}



void shader_core_ctx::rr_top_level_scheduler() {
  shd_warp_t * warp = m_warp[warp_cntr];
  // fprintf(stdout,"RR Top Level scheduler checking Warp %d\n",warp_cntr);
  // // fprintf(stdout,"Top Level scheduler checking Warp %d\n",warp_cntr);
  if (warp->get_elected_status()) {
    bool pushed = push_scalar_que(warp->get_elected_thread());
    
    if (pushed) {
      // fprintf(stdout,"Scalarized thread %d in warp %d\n",warp->get_elected_thread().m_tid, warp->get_elected_thread().m_warp_id);
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

void shader_core_ctx::return_fsm_cycle() {
  for (int warp_id = 0; warp_id < SCALAR_BANDWIDTH; warp_id++) {
    curr_state[warp_id] = next_state[warp_id]; 
    return_fsm_update(warp_id); 
  }
}

void shader_core_ctx::return_fsm_update(int warp_id) {
  next_state[warp_id] = curr_state[warp_id]; 

  switch(curr_state[warp_id]) {
    case IDLE:
      // printf("Core %u, warp %u is in IDLE state\n", get_core_type(), warp_id); 
      if (reconverge_start.test(warp_id) && !reconverge_done.test(warp_id)) {
          next_state[warp_id] = PULL; 
      }
      break; 
    case PULL:
      printf("Core %u, warp %u is in PULL state\n", get_core_type(), warp_id); 
      if (!m_written_register_board->pendingWrites(warp_id)) { // If WRB is empty, go back to IDLE
          printf("Core %u, warp %u has nothing to pull \n", get_core_type(), warp_id); 
          next_state[warp_id] = IDLE; 
          reconverge_done.set(warp_id); 
          wrb_its[warp_id] = m_written_register_board->get_regtable(warp_id).begin(); 
      } else {
          printf("Core %u, warp %u, reg %u pulled from written register board\n", get_core_type(), warp_id, *(wrb_its[warp_id])); 
          next_state[warp_id] = READING;
      }
      break; 
    case READING:   
      printf("Core %u, warp %u is in READING state and will read reg %u\n", get_core_type(), warp_id, *(wrb_its[warp_id])); 
      printf("Reg value is tbd\n");
      // Use wid and regnum to read from register file
      // Put Hunter's code here
      next_state[warp_id] = LOOKUP; 
      break;
    case LOOKUP:
      printf("Core %u, warp %u is in LOOKUP state\n", get_core_type(), warp_id); 
      printf("Goes to SIMT wid=%u, tid=%u\n", div_tid_table->get_entry(warp_id).simt_wid,  div_tid_table->get_entry(warp_id).simt_tid); 

      // Reference divergent thread ID table to find where this belongs in SIMT core
      next_state[warp_id] = SEND;
      break; 
    case SEND:
      printf("Core %u, warp %u is in SEND state\n", get_core_type(), warp_id); 

      // Set values to send to SIMT core
      // Put Hunter's code here

      if ((wrb_its[warp_id]) != m_written_register_board->get_regtable(warp_id).end()) {
        m_written_register_board->releaseRegister(warp_id, *(wrb_its[warp_id]++)); 
        next_state[warp_id] = PULL; 
      } else { // Once all regs in WRB were processed, ready to free the warp and IDLE the return fsm
        reconverge_done.set(warp_id); 
        next_state[warp_id] = IDLE; 
        wrb_its[warp_id] = m_written_register_board->get_regtable(warp_id).begin(); 
        // m_fetched_register_board->reset(); // Reset FRB for the next assigned warp
      }
      break; 
  }
}

void shader_core_ctx::reset_transfer_structures() {
    steal = false;
    for (int i = 0; i < SCALAR_BANDWIDTH; i++) {
      reconverge[i] = false;
      curr_state[i] = IDLE;
      next_state[i] = IDLE; 
      wrb_its[i] = m_written_register_board->get_regtable(i).begin(); // Start going through all regs in WRB
    }
}

bool shd_warp_t::check_at_least_one_on_scalar(shader_core_ctx * m_shader){//if there is something on the scalar queue or the div_tid_table, its means that the scalar core is active
  return !m_shader->get_cluster()->divergent_tid_tables.empty() || !m_shader->get_cluster()->scalar_ques.empty();
}
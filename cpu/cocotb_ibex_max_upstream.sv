// Copyright lowRISC contributors.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// "Maximum" Ibex configuration variant instantiated against a FRESH vendor
// pull of lowRISC/ibex master (see cpu/src_upstream/_UPSTREAM_PROVENANCE.txt
// for the pinned commit), mirroring cocotb_ibex_max.sv's parameter values
// 1:1 so the only thing that differs between the two builds is RTL vintage.
// Extra parameters/ports that exist upstream but not in the old vendored
// tree (RV32ZC, ram_cfg_icache_*, mcounteren_writable_i, lockstep/shadow-core
// signals) are set to their fullest-feature / RTL-default values below.
// Build with: make -f Makefile.upstream CONFIG=max_upstream

module cocotb_ibex(
  input  logic                         clk_i,
  input  logic                         rst_ni,

  // Instruction memory interface
  output logic                         instr_req_o,
  input  logic                         instr_gnt_i,
  input  logic                         instr_rvalid_i,
  output logic [31:0]                  instr_addr_o,
  input  logic [31:0]                  instr_rdata_i,

  // Data memory interface
  output logic                         data_req_o,
  input  logic                         data_gnt_i,
  input  logic                         data_rvalid_i,
  output logic                         data_we_o,
  output logic [3:0]                   data_be_o,
  output logic [31:0]                  data_addr_o,
  output logic [31:0]                  data_wdata_o,
  input  logic [31:0]                  data_rdata_i
);
  parameter bit                 SecureIbex               = 1'b1;
  parameter bit                 ICacheScramble           = 1'b1;
  parameter bit                 PMPEnable                = 1'b1;
  parameter int unsigned        PMPGranularity           = 0;
  parameter int unsigned        PMPNumRegions            = 16;
  parameter int unsigned        MHPMCounterNum           = 29;
  parameter int unsigned        MHPMCounterWidth         = 40;
  parameter bit                 RV32E                    = 1'b0;
  parameter ibex_pkg::rv32m_e   RV32M                    = ibex_pkg::RV32MSingleCycle;
  parameter ibex_pkg::rv32b_e   RV32B                    = ibex_pkg::RV32BFull;
  parameter ibex_pkg::rv32zc_e  RV32ZC                   = ibex_pkg::RV32ZcaZcbZcmp;
  parameter ibex_pkg::regfile_e RegFile                  = ibex_pkg::RegFileFF;
  parameter bit                 BranchTargetALU          = 1'b1;
  parameter bit                 WritebackStage           = 1'b1;
  parameter bit                 ICache                   = 1'b1;
  parameter bit                 DbgTriggerEn             = 1'b1;
  parameter bit                 ICacheECC                = 1'b1;
  parameter bit                 BranchPredictor          = 1'b1;

  // SecureIbex=1 -> MemECC=1 inside ibex_top (upstream now derives this by
  // default: MemECC=SecureIbex, MemDataWidth=MemECC?32+7:32): the core checks
  // instr_rdata_intg_i / data_rdata_intg_i against instr_rdata_i / data_rdata_i
  // using the inverted SECDED(39,32) code (prim_secded_inv_39_32_dec). The
  // cocotb MemAgent only drives the 32 data bits, so we compute the matching
  // ECC bits here from whatever data it returns — otherwise every fetch/load
  // after the first is flagged as a memory-integrity error
  // (alert_major_bus_o / instr_intg_err) and the core traps to mtvec_base
  // forever after instruction 1.
  logic [38:0] instr_rdata_ecc;
  logic [38:0] data_rdata_ecc;

  prim_secded_inv_39_32_enc u_instr_rdata_ecc_enc (
    .data_i (instr_rdata_i),
    .data_o (instr_rdata_ecc)
  );

  prim_secded_inv_39_32_enc u_data_rdata_ecc_enc (
    .data_i (data_rdata_i),
    .data_o (data_rdata_ecc)
  );

  // Upstream split the old single ram_cfg_i tie-off into separate icache
  // tag/data config req/rsp ports (prim_ram_1p_pkg::ram_1p_cfg_req_t/rsp_t,
  // one per icache way) — tie the inputs to their all-zero default config,
  // same intent as the old '.ram_cfg_i (\'b0)'.
  localparam prim_ram_1p_pkg::ram_1p_cfg_req_t RamCfgReqZero = '0;

  ibex_top_tracing #(
      .SecureIbex      ( SecureIbex       ),
      .ICacheScramble  ( ICacheScramble   ),
      .PMPEnable       ( PMPEnable        ),
      .PMPGranularity  ( PMPGranularity   ),
      .PMPNumRegions   ( PMPNumRegions    ),
      .MHPMCounterNum  ( MHPMCounterNum   ),
      .MHPMCounterWidth( MHPMCounterWidth ),
      .RV32E           ( RV32E            ),
      .RV32M           ( RV32M            ),
      .RV32B           ( RV32B            ),
      .RV32ZC          ( RV32ZC           ),
      .RegFile         ( RegFile          ),
      .BranchTargetALU ( BranchTargetALU  ),
      .ICache          ( ICache           ),
      .ICacheECC       ( ICacheECC        ),
      .WritebackStage  ( WritebackStage   ),
      .BranchPredictor ( BranchPredictor  ),
      .DbgTriggerEn    ( DbgTriggerEn     ),
      .DmHaltAddr      ( 32'h00100000     ),
      .DmExceptionAddr ( 32'h00100000     )
  ) u_top (
    .clk_i,
    .rst_ni,

    .test_en_i              ('b0),
    .scan_rst_ni            (1'b1),
    .ram_cfg_icache_tag_i   ('{default: RamCfgReqZero}),
    .ram_cfg_icache_tag_o   (),
    .ram_cfg_icache_data_i  ('{default: RamCfgReqZero}),
    .ram_cfg_icache_data_o  (),

    .hart_id_i              (32'b0),
    // First instruction executed is at 0x0 + 0x80
    .boot_addr_i            (32'h00100000),

    .instr_req_o,
    .instr_gnt_i,
    .instr_rvalid_i,
    .instr_addr_o,
    .instr_rdata_i,
    .instr_rdata_intg_i     (instr_rdata_ecc[38:32]),
    .instr_err_i            (1'b0),

    .data_req_o,
    .data_gnt_i,
    .data_rvalid_i,
    .data_we_o,
    .data_be_o,
    .data_addr_o,
    .data_wdata_o,
    .data_wdata_intg_o      (),
    .data_rdata_i,
    .data_rdata_intg_i      (data_rdata_ecc[38:32]),
    .data_err_i             (1'b0),

    .irq_software_i         (1'b0),
    .irq_timer_i            (1'b0),
    .irq_external_i         (1'b0),
    .irq_fast_i             (15'b0),
    .irq_nm_i               (1'b0),

    .scramble_key_valid_i   ('0),
    .scramble_key_i         ('0),
    .scramble_nonce_i       ('0),
    .scramble_req_o         (),

    .debug_req_i            ('b0),
    .crash_dump_o           (),
    .double_fault_seen_o    (),

    .fetch_enable_i          (ibex_pkg::IbexMuBiOn),
    .mcounteren_writable_i   (ibex_pkg::IbexMuBiOn),
    .alert_minor_o           (),
    .alert_major_internal_o  (),
    .alert_major_bus_o       (),
    .core_sleep_o            (),

    .lockstep_cmp_en_o       (),

    .data_req_shadow_o       (),
    .data_we_shadow_o        (),
    .data_be_shadow_o        (),
    .data_addr_shadow_o      (),
    .data_wdata_shadow_o     (),
    .data_wdata_intg_shadow_o(),

    .instr_req_shadow_o      (),
    .instr_addr_shadow_o     ()
  );
endmodule

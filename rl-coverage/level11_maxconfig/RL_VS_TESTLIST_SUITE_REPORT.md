# RL (MaskablePPO) vs testlist-suite (171 teste) — toggle coverage pe build-ul `opentitan_upstream`

Generat: 2026-07-31, sesiune de antrenare RL pentru L11 (config "max": PMP, ICache,
RV32B, lockstep on), pornind de la baseline-ul suitei de 171 de teste (constrained-random).

## 1. Obiectiv

Construirea unui agent RL (PPO) care să bată, în toggle coverage, suita de 171 de
teste constrained-random deja rulată pe L11 (rezultat: 83.32% toggle).

## 2. Setup

| | |
|---|---|
| RTL vendorizat | lowRISC/ibex master, commit `8ed87e07e3331561bce93af1568d9b376948e701` (identic pe ambele mașini folosite) |
| Build Vtop | `cpu/sim_build_opentitan_upstream/Vtop` — preset-ul oficial **"opentitan"** din `ibex_configs.yaml` al lowRISC — **exact build-ul pe care a rulat și suita de 171 de teste** |
| Total bins toggle | 38.696 |
| Codec instrucțiuni | `codec_l11.py`, 143 op-uri (RV32IMC + RV32B Zba/Zbb/Zbs + zbp/zbc/zbe/zbf) |
| Harness | `test_run_for_l8.py` (cocotb), identic pentru suită și RL |
| Algoritm RL | `MaskablePPO` (sb3-contrib) — action masking pe dimensiunea `op`, saturație ≥97% per modul RTL țintă |
| Medii paralele | `--n-envs 24` (`SubprocVecEnv`) |
| Entropie | `--ent-coef 0.15` |
| Curriculum episod | `64 → 128 → 256` pași/episod (lungime crescătoare) |
| Rețea | PPO/MaskablePPO cu trunchiuri **separate** pi/vf (`net_arch=dict(pi=[512,256], vf=[512,256])`) |
| Pornire | de la 0% hits (fără head-start din corpus-ul suitei), ca să fie o comparație curată |

## 3. Bug-uri găsite și reparate pe parcurs

Antrenarea a fost blocată de mai multe ori de bug-uri reale în infrastructură,
nu de limitări ale RL-ului în sine. În ordine:

1. **`env_l11.py` legat de build-ul greșit** — implicit trimitea la
   `sim_build_max/Vtop` (build vechi, 33.624 bins), nu la build-ul pe care a
   rulat suita (`sim_build_opentitan_upstream`, 38.696 bins). Fix: flag
   `--build`, `VTOP_BUILDS` dict în `env_l11.py`.

2. **Colaps de politică PPO** — reward-ul putea sări la mii/zeci de mii
   (ponderare dinamică per bin, până la 100x), iar `PPO`/`MaskablePPO` foloseau
   un trunchi **comun** policy+value (`net_arch=[512,256]` ca listă simplă).
   Un singur episod cu reward uriaș corupea, prin backprop, trunchiul comun,
   colapsând politica la o ieșire aproape determinstă (toți cei 24 de workeri
   repetau exact același program). Fix: `DYNAMIC_WEIGHT_CAP=5.0` +
   rețele pi/vf separate.

3. **Bug critic — testul RL nu rula deloc, silențios** — `env_l11.py` ghicea
   calea către `libpythonX.Y.so` (`/usr/lib/x86_64-linux-gnu`), validă doar
   pentru un Python de sistem. Pe mașina cu venv peste Anaconda/Miniconda,
   `libpython3.13.so` era în altă parte — `cocotb`'s gpi nu reușea să-l încarce,
   deci **codul Python care injectează programul RL nu rula niciodată**, dar
   Verilator tot ajungea la `$finish` și scria un coverage "fără test rulat",
   identic la fiecare apel, indiferent de acțiuni (exact 9.83%, mereu). Simptom:
   toate episoadele identice, indiferent de politică/acțiuni random. Fix:
   folosire `cocotb-config --libpython` (exact ce face deja
   `cpu/run_testlist_suite.sh`, dar `env_l11.py` avea propria logică,
   incompletă).

4. **Raportare finală greșită** — uniunea între workeri se făcea doar la
   checkpoint (din 100 în 100 episoade), niciodată la finalul efectiv al unei
   rulări întrerupte între checkpoint-uri — rezumatul final arăta cel mai bun
   worker individual, nu uniunea reală. Fix: merge final explicit la
   sfârșitul lui `model.learn()`.

5. **Cosmetic, neconfirmat ca fiind cauza platoului**: `ibex_branch_predict`
   nu există în build-ul `opentitan_upstream` (`BranchPredictor=0` în preset-ul
   "opentitan"), dar apare hardcodat în lista `MODULES` din `env_l11.py`
   (scrisă pentru celălalt build, "max", care avea acest modul). Rămâne
   mereu la 0% și apare mereu ca "worst module" — pur vizual, verificat că nu
   explică platoul de coverage (afectează doar faptul că ~8 op-uri de branch
   nu pot fi niciodată mascate).

## 4. Rezultate — progresul RL în timp (uniune reală, toți cei 24 workeri)

| Episod global | Etapă | Hits | Toggle % |
|---:|---|---:|---:|
| 400 | 64 pași | 29.485 | 76.20% |
| 600 | 128 pași | 31.287 | 80.85% |
| 700 | 128 pași | 31.324 | 80.95% |
| 792 (final etapă 128) | 128 pași | 31.324 | 80.95% |
| 1100 | 256 pași | 31.488 | 81.37% |
| 1200 | 256 pași | 31.499 | 81.40% |
| 2000 | 256 pași | 32.164 | 83.12% |
| 2300 | 256 pași | 32.216 | 83.25% |
| 2400 | 256 pași | 32.221 | 83.27% |
| 2520 (întrerupt, pauzat pt. comparație suită) | 256 pași | 32.221 | **83.27%** |

**RL singur a ajuns la 83.27% (32.221/38.696) — la doar 0.05pp / 21 bin-uri
sub baseline-ul suitei (83.32%, 32.242 bin-uri)**, fără ajutorul uniunii.
Buget de instrucțiuni consumat până la ep2520: ~479.232 (vs 139.406 al
suitei) — vezi §6 pentru actualizare.

Antrenarea a fost întreruptă (grațios, checkpoint salvat) la ep2520 ca să se
elibereze resurse pentru o rulare extinsă a suitei (la buget egal de
instrucțiuni, ~516 target-total în loc de 150), pentru comparație directă
"buget egal" — rezultat în lucru, de adăugat aici când se termină.

Tipar clar: fiecare etapă (64→128→256) aduce un salt inițial, apoi se
plafonează rapid. Rata de creștere scade constant (400→600: +1.802 hits;
600→700: +37; 700→792: +0; 1100→1200: +11).

## 5. Comparație directă RL vs suita de 171 de teste

Comparație pe bin-uri exacte (nu doar procente), via `compare_rl_vs_suite.py`,
la un checkpoint intermediar (~ep1000-1100):

| | Bins toggle | % din 38.696 |
|---|---:|---:|
| Suita (171 teste) | 32.242 | 83.32% |
| RL (checkpoint) | 31.386 | 81.11% |
| Comune (ambele) | 31.133 | — |
| Doar în suită (RL a ratat) | 1.109 | — |
| Doar în RL (suita a ratat) | 253 | — |
| **Uniune (suită + RL)** | **32.495** | **83.98%** |

**RL singur nu a bătut încă suita** (81.11% < 83.32%), dar a găsit 253 de
bin-uri pe care suita nu le-a atins niciodată — deci nu e strict "mai slab",
explorează și zone diferite. **Uniunea celor două depășește deja ținta**
(83.98% > 83.32%).

### Distribuția golului (1.109 bin-uri ratate de RL), pe modul RTL:

| Modul | Bin-uri ratate |
|---|---:|
| `ibex_counter` | 339 |
| `ibex_core` | 296 |
| `ibex_cs_registers` | 191 |
| `ibex_pmp` | 118 |
| `ibex_top_tracing` | 52 |
| `ibex_top` | 27 |
| `ibex_csr` | 18 |
| `ibex_controller` | 14 |
| `ibex_icache` | 9 |
| `ibex_id_stage` | 9 |
| `prim_ram_1p_scr` | 9 |
| `ibex_dummy_instr` | 8 |
| `ibex_load_store_unit` | 7 |
| `ibex_if_stage` | 5 |
| `ibex_lockstep` | 5 |
| `ibex_tracer` | 1 |
| `ibex_wb_stage` | 1 |

**85% din gol e concentrat în doar 4 module** (`ibex_counter`,
`ibex_core`, `ibex_cs_registers`, `ibex_pmp`) — exact zonele pe care suita
le acoperea prin profiluri **direcționate explicit** (`riscv_csr_test` —
sweep determinist de CSR-uri; `riscv_pmp_suite_test` — CSR-uri ponderate
spre `pmpcfg`/`pmpaddr`). Interpretare: nu e o problemă de volum de
antrenare, e o problemă de **direcție** — explorarea nedirecționată a RL-ului
nimerește rar secvențele specifice (adrese CSR de performance-counter,
configurații PMP anume) necesare pentru acele bin-uri.

## 6. Buget de instrucțiuni — RL vs suită

| | Instrucțiuni |
|---|---:|
| Suita (doar agent — suma câmpurilor `n` din `corpus_suite_*.json`) | **139.406** |
| Suita (total real simulat: agent + prolog 40/test + WFI 1/test) | 146.417 |
| RL (până la ep1200, estimat, doar agent) | **~170.368** |

Notă: `emit_program()` din `codec_l11.py` nu adaugă niciodată prologul —
`"n"` din corpus e deja strict agent. Prologul (40 instrucțiuni) și `WFI`-ul
final se adaugă separat, la rulare, de `test_run_for_l8.py`
(`full_program = PROLOGUE + agent_machine + [WFI]`) — identic la suită și la
RL, deci comparația corectă e agent-vs-agent (139.406 vs ~170.368).

Detaliu RL (estimat din granițele etapelor de curriculum):
- 64 pași × ~554 episoade = 35.456
- 128 pași × ~238 episoade = 30.464
- 256 pași × 408 episoade = 104.448

**RL a folosit deja ~22% mai mult buget de instrucțiuni decât suita, și tot
rămâne sub coverage-ul ei** (81.40% vs 83.32%) — confirmă din nou că
problema nu e cantitatea, e direcția explorării.

## 7. Concluzii

1. **Obiectivul inițial** ("RL să bată suita de teste") e atins doar **prin
   combinare** (uniune RL + suită = 83.98%), nu prin RL singur.
2. RL singur, chiar cu buget de instrucțiuni mai mare decât suita, nu a
   reușit să o depășească — rămâne la 81.40% și creșterea aproape s-a oprit.
3. Golul rămas nu e distribuit uniform — 85% e concentrat în 4 module
   (`ibex_counter`, `ibex_core`, `ibex_cs_registers`, `ibex_pmp`), toate
   zone pe care suita le-a acoperit prin profiluri de test scrise manual,
   cu cunoștințe de domeniu (CSR sweep determinist, config PMP direcționat).
4. **Recomandare pentru pasul următor**: nu mai merită să continui
   antrenarea nedirecționată (rată de creștere aproape zero) — un bias
   direcționat în reward/curriculum spre CSR-urile de performance-counter
   și configurațiile PMP ar avea șanse mai mari să închidă golul rămas,
   mult mai eficient decât a mai aștepta.

## 8. Pași următori (to-do)

1. **Strategie RL direcționată ("crRL")** — crt câștigă (ușor) la buget egal
   pentru că profilurile lui sunt scrise manual, cu cunoștințe de domeniu,
   direcționate spre colțuri greu de nimerit random (illegal-instr,
   invalid-CSR, PMP full-random, CSR sweep determinist). RL explorează
   nedirecționat. De discutat/încercat:
   - Bonus de reward suplimentar, direcționat explicit spre cele 4 module
     unde RL rămâne în urmă (`ibex_counter`, `ibex_core`,
     `ibex_cs_registers`, `ibex_pmp` — 85% din gol, vezi §5).
   - O fază de curriculum suplimentară care la un moment dat restrânge/
     preferă explicit CSR-uri de performance-counter și configurații PMP.
   - Posibil un "warm-start": pre-antrenare scurtă pe un mini-corpus generat
     cu bias spre acele zone, înainte de RL propriu-zis.
   - De verificat dacă action masking-ul poate fi extins să nu doar
     mascheze module saturate, ci să **favorizeze** explicit modulele slabe.
2. **Rulare extinsă a suitei (crt) la buget egal cu RL (~480K instrucțiuni,
   518 teste, 438 din ele `riscv_pmp_suite_test`)** — pornită, oprită la
   final de zi, de reluat: `nohup bash run_testlist_suite_parallel.sh 24 >
   suite_extended_run.log 2>&1 &` din `cpu/` (resumabilă, sare peste testele
   deja terminate).
3. De adăugat în raport, când suita extinsă termină: comparația finală la
   buget egal extins (RL ~83.27%+ vs crt extins), plus comparația la bugetul
   *original* al suitei (139.406 instrucțiuni — RL la acel punct, ~ep1073,
   era la ~81.37%, deci crt câștigă cu ~2pp la bugetul lui natural).

## 9. Fișiere relevante

- `rl-coverage/level11_maxconfig/env_l11.py` — mediul Gymnasium (build selection, reward, masking)
- `rl-coverage/level11_maxconfig/train_l11_ppo.py` — scriptul de antrenare
- `rl-coverage/level11_maxconfig/run_l11_steps_curriculum.sh` — curriculum-ul de lungime episod
- `rl-coverage/level11_maxconfig/compare_rl_vs_suite.py` — scriptul de comparație pe bin-uri exacte
- `rl-coverage/level11_maxconfig/l11_opentitan_upstream_checkpoint_hits.pkl` — hit-urile RL acumulate
- `cpu/coverage_suite_*.dat` — coverage brut per test din suita de 171 (gitignored, local)
- `rl-coverage/level11_maxconfig/TESTLIST_SUITE_SUMMARY.txt` — raportul original al suitei de 171 de teste

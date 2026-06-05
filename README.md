# Phase-Belief LIBERO

Offline phase-belief experiments on LIBERO-GOAL trajectories.

This project studies whether a phase-belief module can discover stable latent motion phases from low-dimensional robot trajectories.

Current main setting:

- Dataset: LIBERO-GOAL, 10 HDF5 task files, around 500 successful demos
- Split: demo-level train / val / test split
- Input: low-dimensional robot state and action history
- No image input in the current version
- No privileged simulator `states` used for training

## Current main conclusion

In the current low-dimensional GRU setting, the key mechanism is:

`phase bottleneck + L_action + L_conf + L_balance`

where:

- phase bottleneck prevents the predictor from bypassing the phase variable
- `L_conf` makes each timestep's phase belief low-entropy and clear
- `L_balance` prevents phase collapse
- `L_state` does not significantly improve phase clarity in the current low-dimensional setting
- `L_smooth` is not necessary in the current best phase-discovery model

The current preferred phase-discovery model is:

`action_conf_balance`

## Directory structure

```text
phase_belief/
  data/
    dataset.py
  models/
    phase_belief_bottleneck_gru.py
    ...

tools/
  data/
    build_demo_split_index.py
    download_libero_goal_missing.py
    check_hdf5_demo.py

  train/
    train_lowdim_phase_bottleneck.py

  eval/
    eval_checkpoint_on_index.py
    eval_phase_split_demos.py

  analysis/
    compare_short_segments.py
    make_phase_viz_report.py
    visualize_phase_demo_bottleneck.py

  debug/
    test_dataset.py

  legacy/
    build_window_index_window_split.py
    train_lowdim_phase.py
    visualize_phase_demo.py
    eval_phase_batch.py

Main workflow
1. Build demo-level split index
python tools/data/build_demo_split_index.py \
  --data-dir /root/autodl-tmp/LIBERO/datasets/libero_goal \
  --out-dir /root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_10files_demo_split \
  --seq-len 16 \
  --pred-len 4 \
  --train-ratio 0.70 \
  --val-ratio 0.15 \
  --seed 42
2. Train current main model: action_conf_balance
python tools/train/train_lowdim_phase_bottleneck.py \
  --train-index /root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_10files_demo_split/train_index.json \
  --val-index /root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_10files_demo_split/val_index.json \
  --save-dir /root/autodl-tmp/phase_belief_libero/checkpoints/ablation_10files/action_conf_balance \
  --epochs 30 \
  --batch-size 256 \
  --num-phases 4 \
  --hidden-dim 128 \
  --lambda-state 0.0 \
  --lambda-smooth 0.0 \
  --lambda-balance 0.02 \
  --lambda-conf 0.01
3. Evaluate checkpoint on test index
python tools/eval/eval_checkpoint_on_index.py \
  --ckpt /root/autodl-tmp/phase_belief_libero/checkpoints/ablation_10files/action_conf_balance/best.pt \
  --index /root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_10files_demo_split/test_index.json \
  --out /root/autodl-tmp/phase_belief_libero/outputs/ablation_10files/action_conf_balance/test_metrics.json
4. Evaluate test-demo phase segments and visualization
python tools/eval/eval_phase_split_demos.py \
  --ckpt /root/autodl-tmp/phase_belief_libero/checkpoints/ablation_10files/action_conf_balance/best.pt \
  --split-records /root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_10files_demo_split/split_demo_records.json \
  --split test \
  --out-dir /root/autodl-tmp/phase_belief_libero/outputs/ablation_10files/action_conf_balance_phase_eval \
  --seq-len 16 \
  --max-plots-per-task 3
5. Generate HTML visualization report
python tools/analysis/make_phase_viz_report.py \
  --plot-dir /root/autodl-tmp/phase_belief_libero/outputs/ablation_10files/action_conf_balance_phase_eval/plots \
  --out /root/autodl-tmp/phase_belief_libero/outputs/ablation_10files/action_conf_balance_phase_eval/phase_viz_report.html \
  --pattern "*.png"
6. Compare short segment ratio between full and action_conf_balance
python tools/analysis/compare_short_segments.py \
  --full /root/autodl-tmp/phase_belief_libero/outputs/10files_demo_split_phase_eval/test_segments.jsonl \
  --acb /root/autodl-tmp/phase_belief_libero/outputs/ablation_10files/action_conf_balance_phase_eval/test_segments.jsonl \
  --out-dir /root/autodl-tmp/phase_belief_libero/outputs/short_segment_compare_10files \
  --thresholds 1 2 3 5
Notes

Do not commit datasets, checkpoints, generated outputs, or logs.

Expected ignored data locations:

/root/autodl-tmp/LIBERO/datasets
/root/autodl-tmp/phase_belief_libero/checkpoints
/root/autodl-tmp/phase_belief_libero/outputs
/root/autodl-tmp/phase_belief_libero/data_indices
/root/autodl-tmp/phase_belief_libero/logs

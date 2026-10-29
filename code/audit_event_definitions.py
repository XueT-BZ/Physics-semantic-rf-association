"""Export the 14 latent transmissions in nominal test scene 0, without refitting.

The prefix below mirrors experiment.simulate's random draws through the collision
override. The original simulator is left unchanged so previous results retain
their recorded code hash. Event parameters are evaluation-only provenance.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from experiment import Config, simulate, scene_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    seed = 1_020_000
    cfg = Config()
    rng = np.random.default_rng(seed)
    S, E = cfg.receivers, cfg.events
    rng.uniform(0, 2*np.pi)  # ring rotation
    rng.normal(0, cfg.clock_us*1e-6, S)
    rng.normal(0, cfg.freq_bias_khz*1e3, S)
    tau = rng.uniform(0, cfg.window_us*1e-6, E)
    channel = rng.integers(0, 6, E)
    eta = rng.normal(0, 180, E)
    carrier = 100e6 + channel*1800 + eta
    bw = rng.uniform(20e3, 60e3, E)
    dur = rng.uniform(20e-6, 80e-6, E)
    loc = rng.uniform(-1500, 1500, (E, 2))
    protocol = rng.integers(0, 4, E)
    parent = np.full(E, -1)
    for e in range(1, E):
        if rng.random() < cfg.collisions:
            a = int(rng.integers(0, e))
            parent[e] = a
            tau[e] = tau[a] + rng.normal(0, 1.5e-6)
            carrier[e] = carrier[a] + rng.normal(0, 150)
            bw[e] = max(1e3, bw[a] + rng.normal(0, 2e3))
            dur[e] = max(1e-6, dur[a] + rng.normal(0, 2e-6))
    scene = simulate(seed, cfg)
    saved = pd.read_csv(ROOT/'results/observations_r00_test.csv.gz', float_precision='round_trip')
    saved = saved[saved.scene == 0].reset_index(drop=True)
    expected = scene_csv(scene, 0, 'test', 0)
    pd.testing.assert_frame_equal(saved[expected.columns], expected, check_dtype=False,
                                  rtol=1e-12, atol=1e-14)
    receivers = [scene.obs['receiver'][scene.truth == e].tolist() for e in range(E)]
    assert all(len(x) == len(set(x)) for x in receivers)
    ledger = pd.DataFrame({
        'event_id': np.arange(E), 'scene_seed': seed,
        'emission_time_us': tau*1e6, 'carrier_hz': carrier,
        'bandwidth_khz': bw/1e3, 'duration_us': dur*1e6,
        'source_x_m': loc[:, 0], 'source_y_m': loc[:, 1],
        'anonymous_class': [f'Q{q+1}' for q in protocol],
        'collision_parent_event_id': parent,
        'initial_center_index_before_override': channel,
        'initial_carrier_perturbation_hz_before_override': eta,
        'observed_receiver_ids': [','.join(map(str, sorted(x))) for x in receivers],
        'observed_reports': list(map(len, receivers)),
    })
    ledger.to_csv(ROOT/'results/event_ledger_r00_test000.csv', index=False)
    report = {
        'status': 'PASS', 'scene_seed': seed, 'latent_events': E,
        'genuine_reports': int(sum(map(len, receivers))),
        'clutter_reports': int(scene.clutter.sum()),
        'carrier_center_spacing_hz': 1800, 'carrier_event_perturbation_sd_hz': 180,
        'initial_bandwidth_range_hz': [20000, 60000],
        'nominal_receiver_oscillator_sd_hz': 250,
        'carrier_measurement_sd_hz_by_receiver': [120, 200, 280, 360],
        'same_event_shared_carrier_cancels_from_pair_difference': True,
        'event_labels_are_anonymous_not_named_protocols': True,
        'regenerated_observations_match_delivered_csv': True,
        'experiment_sha256': hashlib.sha256((ROOT/'code/experiment.py').read_bytes()).hexdigest(),
        'scope': 'One scene provenance audit; no new performance experiment or RF measurement.'
    }
    target = ROOT/'audit/revision_20261020/event_definitions.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(ledger[['event_id', 'anonymous_class', 'carrier_hz', 'bandwidth_khz',
                  'duration_us', 'observed_receiver_ids']].to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

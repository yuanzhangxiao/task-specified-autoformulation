# Delay SINDy autoencoder baseline

A latent equation-discovery method from the literature, run on the Phase C
CSTR-hard cell (`phase_c_cstr_controlled_reactor_mechanism_canonical_named_hard_reset_v1`)
so that the reactor intervention study can score it with the other baselines.
Code: `autoformalism.baselines.sindy_autoencoder`; runner:
`scripts/run_sindy_autoencoder_baseline.py`; frozen plan:
`configs/cstr_hard_sindy_autoencoder_v1.json`.

## Method

The SINDy autoencoder (Champion, Lusch, Kutz and Brunton, PNAS 2019) learns at
once an encoder from the measurements to a few latent coordinates, a decoder
back, and a sparse polynomial model of the latent coordinates' derivative. With
one measured quantity, the deep delay autoencoder (Bakarji, Champion, Kutz and
Brunton, Proc. R. Soc. A 2023) feeds the encoder time-delay coordinates of it,
asks the first latent coordinate to equal the measurement, and integrates the
SINDy model across the delay window for a consistency loss. The reactor's
temperature is the one measured quantity; its three inputs are measured.

    h_k = [y_k, y_(k-1), ..., y_(k-n+1)],   z_k = phi(h_k),   z_k1 ~ y_k
    dz/dt = Xi^T theta([z, u(t)])

Taken from the papers and the authors' code
(`josephbakarji/deep-delay-autoencoder`, commit 1f43e54):

- the six loss terms of their equation 15 (reconstruction, the SINDy losses in
  `dh/dt` and `dz/dt` by automatic differentiation, the first-coordinate loss,
  the RK4 consistency loss across the window, clipped at 500 as in the code, and
  the L1 penalty) with the weights of the code's real-data case
  (`testcases/lorenzww_basic.py`): 0.3, 0.001, 0.001, 0.01, 0.1 and 1e-5;
- `dh/dt` from finite differences of each record (`np.gradient`, as the code's
  `RealData`); no smoothing or interpolation;
- encoder widths 64 and 32 (Champion et al.'s Lorenz network), mirrored in the
  decoder, ELU activations (the delay code's), Xavier-uniform weights;
- a degree-2 polynomial library (the delay code's real-data case);
- Adam, learning rate 0.001, mini-batches of 32 (the delay code);
- 300 epochs (the delay code's real-data case) with sequential thresholding at
  0.1 (Champion et al.) every 100 epochs (the delay code's frequency), then 100
  refinement epochs without the L1 penalty and with the sparsity pattern fixed
  (Champion et al.'s refinement).

Adapted for this protocol:

- **Inputs.** The library holds the measured inputs, as in SINDy with control
  (Brunton, Proctor and Kutz 2016): `theta` is the degree-2 polynomial of
  `[z, u]`. The inputs are centred and divided by their largest training
  deviation, as latent-state SINDy scales them; the temperature is standardized.
- **A causal window, at rest before the record.** The papers' windows run
  forward from the encoded time; a forecast from the first reading cannot read
  ahead, so the window ends at the encoded time and samples before a record's
  first are filled with the first reading. Before every training record but one,
  and before 22 of the 24 held-out trajectories of the combined suite, the
  reactor was at rest at its first reading, so the filled window is the true
  history; the others start away from the operating point and the filled window
  reads as rest at the first reading, the only information any model receives.
  A forecast is `z(t0) = phi([y0, ..., y0])`, then the SINDy model integrated with
  the inputs; its first coordinate is the temperature. The consistency loss
  integrates from each encoded time across the next `n - 1` samples.
- **Starting coefficients.** `Xi` starts at N(0, 0.01^2), as latent-state SINDy's.
  The papers' starts are on their systems' scales (all ones in Champion et al.;
  N(0, 10^2) in the delay code). In a 30-epoch feasibility check on the training
  data, the all-ones start left the first latent coordinate away from the
  temperature (first-coordinate loss 1.8 against 0.0005) and kept every
  coefficient above the threshold. One full-length task (latent dimension 3,
  window 10, seed 0) then confirmed that training stays finite and that its
  validation rollouts are finite; the settings were written before it finished
  and none was changed after it. No setting was chosen from a validation or test
  score.
- **Sizes and selection.** The papers fix the latent dimension and window from
  knowledge of their systems. Here the latent dimension (2, 3 or 4, temperature
  included, as latent-state SINDy's 1 to 3 added coordinates) and the window (5,
  10 or 20 samples, 0.5 to 2 time units, around the hidden lag of 0.2 to 0.6)
  are chosen per seed by validation free-rollout error in the adaptive replay,
  and validation free rollouts choose the refinement checkpoint, as for the
  other latent baselines.

The latent coordinates other than the first name no physical quantity, so the
declared-mechanism scoring cannot read a feed, reaction or jacket term from the
equations.

## Commands

    PYTHONPATH=src python scripts/run_sindy_autoencoder_baseline.py plan --root <root> \
        --release <phase-c-development-final-v2> --config configs/cstr_hard_sindy_autoencoder_v1.json
    PYTHONPATH=src python scripts/run_sindy_autoencoder_baseline.py train --root <root> --index <0-26>
    PYTHONPATH=src python scripts/run_sindy_autoencoder_baseline.py select --root <root>

Twenty-seven tasks (3 seeds x 3 latent dimensions x 3 windows), about five
minutes each on one CPU thread.

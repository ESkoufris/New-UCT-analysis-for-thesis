# --- Setup ---
include("../../MCTSUtils.jl")
using .MCTSUtils
using Plots
using Random
using LaTeXStrings
using Measures
using Printf

const OUTDIR = "new_figures/UCT/500k"
mkpath(OUTDIR)
Random.seed!(1)   # global reproducibility

# --- Fixed params ---
S = 2
γ = 0.8
mcts_iterations = 500_000
n_simulations   = 200
c_param         = 4
verbose_flag    = false
logscale        = "x"      # "none" | "x" | "y" | "xy"

A_list = 2:3
H_list = 3:4

# --- style helpers ---
guide_fnt = font(26, "Computer Modern")
tick_fnt  = font(18, "Computer Modern")
title_fnt = font(40, "Computer Modern", "bold")

# --- All transition/reward cases ---
cases = [
    (true,  true,  "trans_det-rew_det"),
    (true,  false, "trans_det-rew_sto"),
    (false, true,  "trans_sto-rew_det"),
    (false, false, "trans_sto-rew_sto")
]

# Deterministic seed function per (case_index, A)
seed_for = (case_idx, A) -> 123456789 + 10007*A + 1009*case_idx  # stable across runs

for (case_idx, (det_transitions, det_rewards, casetag)) in enumerate(cases)
    for A in A_list
        # One fixed seed per (case, A), reused for all H
        fixed_seed = seed_for(case_idx, A)

        for H in H_list
            # Output dir per case/A/H
            outdir = joinpath(OUTDIR, casetag, "A$(A)", "H$(H)")
            mkpath(outdir)

            # Build MDP with the SAME seed for this (case, A), even if H changes
            means, mdp = random_MDP(
                S, A;
                γ = γ,
                is_deterministic = det_transitions,
                horizon = H,
                seed = fixed_seed,                 # <-- key change
                deterministic_rewards = det_rewards,
                bernoulli_rewards = !det_rewards
            )

            # Run MCTS
            avg_returns, T_samples = run_MCTS(
                mcts_iterations = mcts_iterations,
                n_simulations   = n_simulations,
                mdp             = mdp,
                H               = H,
                verbose         = verbose_flag,
                c_param         = c_param
            )

            # Error + variability band (keep your choice)
            p = compute_regret_at_root_with_variability(
                mdp, avg_returns, T_samples;
                divide_by_log = false,
                logscale = logscale,
                band = :std
                # or: band = (0.25, 0.75)
            )

            plot!(p;
                legend = false,
                linewidth = 4,
                framestyle = :box,
                grid = true,
                guidefont = guide_fnt,
                tickfont  = tick_fnt,
                titlefont = title_fnt,
                xlabel = "Iterations",
                ylabel = "Error",
                title = @sprintf("A = %d   |   H = %d", A, H),
                titlelocation = :center,
                size = (1800, 1200),
                margin = 6mm
            )

            # Save figure
            png_path = joinpath(outdir, @sprintf("UCT_error_A%d_H%d.png", A, H))
            pdf_path = joinpath(outdir, @sprintf("UCT_error_A%d_H%d.pdf", A, H))
            savefig(p, png_path)
            savefig(p, pdf_path)

            @info "Saved plots for $(casetag), A=$(A), H=$(H) → $png_path (seed=$(fixed_seed))"
        end
    end
end

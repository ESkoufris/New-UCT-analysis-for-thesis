# --- Setup ---
include("../../MCTSUtils.jl")
using .MCTSUtils
using Plots
using Random
using LaTeXStrings
using Measures
using Printf

const OUTDIR = "new_figures/UCT"
mkpath(OUTDIR)
Random.seed!(1)   # <-- keep your fixed seed

# --- Fixed params ---
S = 2                      # explicitly fixed
γ = 0.9
mcts_iterations = 100_000
n_simulations   = 300
c_param         = 1
verbose_flag    = false
logscale        = "x"      # "none" | "x" | "y" | "xy"

A_list = 2:3
H_list = 3:4

# --- style helpers ---
guide_fnt = font(26, "Computer Modern")          # axis labels
tick_fnt  = font(18, "Computer Modern")          # tick labels
title_fnt = font(40, "Computer Modern", "bold")  # big centered title

# --- All transition/reward cases ---
cases = [
    (true,  true,  "trans_det-rew_det"),
    (true,  false, "trans_det-rew_sto"),
    (false, true,  "trans_sto-rew_det"),
    (false, false, "trans_sto-rew_sto")
]

for (det_transitions, det_rewards, casetag) in cases
    for A in A_list, H in H_list
        # Output dir per case/A/H
        outdir = joinpath(OUTDIR, casetag, "A$(A)", "H$(H)")
        mkpath(outdir)

        # Random seed per run (store if you write sidecars)
        seed = rand(1:10^9)

        # Build MDP
        means, mdp = random_MDP(
            S, A;
            γ = γ,
            is_deterministic = det_transitions,
            horizon = H,
            seed = seed,
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

        # --- Plot mean error with variability band (NOT SE) ---
        # Option 1: ±1 Standard Deviation (shows variance)
        p = compute_regret_at_root_with_variability(
            mdp, avg_returns, T_samples;
            divide_by_log = false,
            logscale = logscale,
            band = :std
        )

        # Option 2: Quantile band (e.g., IQR). Uncomment to use.
        # p = compute_regret_at_root_with_variability(
        #     mdp, avg_returns, T_samples;
        #     divide_by_log = false,
        #     logscale = logscale,
        #     band = (0.25, 0.75)
        # )

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

        @info "Saved plots for $(casetag), A=$(A), H=$(H) → $png_path"
    end
end
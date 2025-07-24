using Pkg
Pkg.activate("/home/mar9328/VSCodeProjects/CartesianGeneticProgramming.jl")
using MAT
using Dates
using Random
using Dierckx
using Cambrian
using SeisNoise
using Statistics
using NumericalIntegration
using CartesianGeneticProgramming
# include("../model_template_utils.jl")
include("scripts/utils/evaluation_metrics.jl")
include("scripts/utils/dataloader_global.jl")


function evaluate_dataset(ind::CGPInd, point_dataset)
    CartesianGeneticProgramming.reset!(ind)
    _t, _x, _e, _sla, _fw, _lat, _lon = point_dataset
    dt = _t[2] - _t[1]
    local inputs
    
    inputs = Vector{Float64}.([_e, _sla, _fw])

    res = process(ind, inputs)
    dxdts = res[1]

    if dxdts == nothing || length(dxdts) != length(_x)
        return -Inf
    end

    vx = collect(1.0:length(dxdts))
    _X = NumericalIntegration.cumul_integrate(vx, dxdts) .* dt
    
    B = hcat(ones(length(dxdts)), _t, _X)
    Â = B\_x
    pred = B * Â

    score = mielke(_x, pred)

    if isequal(score, nothing) || isnan(score)
        return -Inf
    else
        return score
    end
end

function sample_coastal_point(_ds, index=NaN)
    _t, _X, _EWave, _SLA, _FW, _LAT, _LON = _ds
    
    if isnan(index)
        index = rand(1:length(lon))
    end
    
    _x = _X[:, index]
    _e = _EWave[:, index]
    _sla = _SLA[:, index]
    _fw = _FW[:, index]
    _lat = _LAT[index]
    _lon = _LON[index]

    return _t, _x, _e, _sla, _fw, _lat, _lon
end

function evaluate_individual(ind, eval_indices)
    ### fixed indices every n points
    scores = zeros(length(eval_indices))

    for i in 1:length(eval_indices)
        scores[i] = evaluate_dataset(ind, sample_coastal_point(dataset_allpoints, eval_indices[i]))
    end

    median_score = nanmedian(scores)
    if isnan(median_score) # if all scores are NaN
        median_score = -Inf
    end

    return [median_score]
end

dataset_allpoints = global_dataset(;normalize=true, calibration=true, filter_limits=(9, 120))

cfg_filename = "archive/config.yaml"
outdir = "archive"
n_skip_points = 50 #  interval of coastal points skipped between consecutive training points 
start_seed = 410
end_seed = 419
    
println("seeds: $(collect(start_seed:end_seed))")
for s in collect(start_seed:end_seed)
    Random.seed!(s)
    
    job_name = "GLOBAL-newdataset_3inputs-1outputs-interannual-fixedeval_npoints$(n_skip_points)-run$s"
    cfg = get_config(cfg_filename; n_population=200, id=job_name, n_in=3, n_out=1, columns=30, d_fitness=1, n_offsprings=200, n_gen=2000, log_gen=1, save_gen=1000, output_dir=outdir, n_skip_points=n_skip_points)

    fit(i::CGPInd, indices::Vector{Int64}) = evaluate_individual(i, indices)
    
    test_inputs = [
        dataset_allpoints[INDEX_E][:, 1], 
        dataset_allpoints[INDEX_SLA][:, 1], 
        dataset_allpoints[INDEX_FW][:, 1]
        ]

    CartesianGeneticProgramming.mutate(i::CGPInd) = goldman_mutate(cfg, i, test_inputs, fit)

    logfile = joinpath(cfg.output_dir, "logs", string(cfg.id, ".csv"))
    e = CGPEvolution(cfg, fit; logfile=logfile)
    run!(e)
end

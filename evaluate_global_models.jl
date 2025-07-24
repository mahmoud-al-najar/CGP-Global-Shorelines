using Pkg
Pkg.activate("/home/mar9328/VSCodeProjects/CartesianGeneticProgramming.jl")
using Plots
using JSON
using JLD2
using CSV, DataFrames
using StatsBase
using Clustering
using Distributions
using NumericalIntegration
using NaNStatistics
using CartesianGeneticProgramming
using Distributions
using DataStructures
using GlobalSensitivityAnalysis
include("scripts/utils/evaluation_metrics.jl")
include("scripts/utils/dataloader_global.jl")
include("scripts/plotting/graphs.jl")
include("scripts/plotting/map_plot.jl")
include("scripts/plotting/map_plot_discrete.jl")


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

function evaluate_runs_satellite_dataset(inds, satellite_dataset; interval=1)  
    
    eval_point_indices = collect(1:interval:length(satellite_dataset[INDEX_lon]))

    scores = Array{Float64, 2}(undef, length(inds), length(eval_point_indices))
    preds = Array{Float64, 3}(undef, length(inds), length(eval_point_indices), length(satellite_dataset[INDEX_t][1:END_I]))

    for ind_i in eachindex(inds)
        ind = inds[ind_i]
        if ind_i % 50 == 0
            println("\tind_i: $ind_i")
        end
        
        for point_i in eachindex(eval_point_indices)
            dataset = sample_coastal_point(satellite_dataset, eval_point_indices[point_i])
            _t, _x, _e, _sla, _fw, _lat, _lon = dataset
            
            dt = _t[2] - _t[1]
            local inputs
            
            inputs = Vector{Float64}.([_e, _sla, _fw])
        
            CartesianGeneticProgramming.reset!(ind)
            res = process(ind, inputs)
            dxdts = res[1]

            if dxdts == nothing || length(dxdts) != length(_x)
                println(dxdts == nothing, "    ", length(dxdts), "    ", length(_x))
                scores[ind_i, point_i] = -Inf
            else
                vx = collect(1.0:length(dxdts))
                X = NumericalIntegration.cumul_integrate(vx, dxdts) .* dt
                
                B = hcat(ones(length(dxdts[1:END_I])), _t[1:END_I], X[1:END_I])
                Â = B\_x[1:END_I]
                pred = B * Â
                score = corr(_x[1:END_I], pred; digits=2)

                scores[ind_i, point_i] = score
                preds[ind_i, point_i, :] .= pred
            end
        end
    end
    replace!(scores, NaN=>-Inf)

    return scores, preds
end

gen_heatmap(fits) = Plots.heatmap(map(x->"P-$x", collect(1:size(fits)[2])), map(x->"Ind-$x", collect(1:size(fits)[1])), fits, clim=(0.0, 1.0))
gen_heatmap(fits, xs, ys) = Plots.heatmap(xs, ys, fits, clim=(0.0, 1.0))

### adapt to exps
n_skip_points = 50
eval_n_skip_points = 1

### 3 input runs
evo_output_dir="archive"
plots_output_dir="outputs"
job_name, short_job_name = "GLOBAL-newdataset_3inputs-1outputs-interannual-fixedeval_npoints$(n_skip_points)", "global-3inputs-interannual-nskip$(n_skip_points)"

cfg_filename = "archive/config.yaml"
cfg = get_config(cfg_filename; n_population=200, n_in=3, d_fitness=1, n_offsprings=200, n_gen=2000, log_gen=1, save_gen=1000, n_skip_points=n_skip_points, columns=30)

### evolution plots
logfiles = [lf for lf in readdir("$(evo_output_dir)/logs")]
runs = [split(lf, '-')[end][1:end-4] for lf in logfiles]
log_dfs = []
runs_gens_fits = Array{Float64, 2}(undef, length(runs), Int(cfg.n_gen/cfg.log_gen))
for lf_i in 1:lastindex(logfiles)#[1:1]
    lf = logfiles[lf_i]
    file = "$(evo_output_dir)/logs/$(lf)"
    df = CSV.read(file, DataFrame; header=["timestamp", "cambrian", "level", "gen", "max", "mean", "std"])
    
    push!(log_dfs, df)
    runs_gens_fits[lf_i, :] .= df.max
end

#### load inds per run
run_inds = []
for n_run in runs    
    run_search = cfg.n_gen
    
    gen_path = "$(evo_output_dir)/gens/$(job_name)-$n_run/$(lpad(run_search,4,"0"))"
    while !isdir(gen_path) && run_search > 0
        run_search -= cfg.save_gen
        gen_path = "$(evo_output_dir)/gens/$(job_name)-$n_run/$(lpad(run_search,4,"0"))"
    end
    if ispath(gen_path)
        push!(run_inds, [])
        println("$n_run: $gen_path")
        
        inds = Array{CGPInd}(undef, cfg.n_population)
        
        dna_files = readdir(gen_path)
        for dna_i in 1:lastindex(dna_files)
            dna_file = dna_files[dna_i]
            path_dna = "$gen_path/$dna_file"
            dna = JSON.parsefile(path_dna)
            ind_fit = dna["fitness"]
            chromo = convert(Array{Float64,1}, dna["chromosome"])
            ind = CGPInd(cfg, chromo)
            ind.fitness .= replace!(ind_fit, nothing => -Inf)
            inds[dna_i] = ind
            push!(run_inds[end], ind)
        end
    else
        println("Doesn't exist: $gen_path")
    end
end

println("Loading dataset")
dataset_allpoints = global_dataset(;normalize=true, calibration=false, filter_limits=(9, 120))
n_runs = length(run_inds)

run_scores = Array{Float64, 3}(undef, n_runs, cfg.n_population, length(collect(1:eval_n_skip_points:length(dataset_allpoints[INDEX_lon]))))
run_preds = Array{Float64, 4}(undef, n_runs, cfg.n_population, length(collect(1:eval_n_skip_points:length(dataset_allpoints[INDEX_lon]))), length(dataset_allpoints[INDEX_t][1:END_I]))

run_medianscores = Array{Float64, 2}(undef, n_runs, cfg.n_population)

println("Evaluating all runs")
for run_i in 1:n_runs
    println("> Run: $(run_i)")
    run_scores[run_i, :, :], run_preds[run_i, :, :, :] = evaluate_runs_satellite_dataset(run_inds[run_i], dataset_allpoints; interval=eval_n_skip_points)    
    run_medianscores[run_i, :] = [nanmean(run_scores[run_i, i, :]) for i in 1:cfg.n_population]
end
replace!(run_medianscores, NaN=>-Inf)

### find best from individual runs
# for run_i in 1:n_runs
#     modelorder = sortperm(run_medianscores[run_i, :], rev=true)
#   
#     plot_lons = dataset_allpoints[INDEX_lon][1:eval_n_skip_points:end]
#     plot_lats = dataset_allpoints[INDEX_lat][1:eval_n_skip_points:end]
#     plot_scores = run_scores[run_i, modelorder[1], :]
#     plot_preds = run_preds[run_i, modelorder[1], :, :]'
#
#     plotorder = sortperm(plot_scores)
#
#     maps_plot(plot_lons[plotorder], plot_lats[plotorder], plot_scores[plotorder]; suptitle="$(short_job_name)-run$(run_i)-topmean", score=round(nanmean(plot_scores), digits=2), pdf_path=plots_output_dir, marker_size=3, size=(1280, 640), vmin=0.0, vmax=1.0, extension="png")
# end

### evaluating grouped population (all runs)
all_inds = vcat(run_inds...)
all_fitness = [ind.fitness[1] for ind in all_inds]
all_scores = Array{Float64, 2}(undef, n_runs*cfg.n_population, length(collect(1:eval_n_skip_points:length(dataset_allpoints[INDEX_lon]))))

for run_i in 0:n_runs-1
    all_scores[(run_i*cfg.n_population)+1:(run_i+1)*cfg.n_population, :] .= run_scores[run_i+1, :, :]
end
replace!(all_scores, NaN=>-Inf)

medianscores = [nanmedian(all_scores[i, :]) for i in 1:size(all_inds, 1)]
replace!(medianscores, NaN=>-Inf)

medianorder = sortperm(medianscores, rev=true)

### make heatmaps
p_all_inds_heatmap = gen_heatmap(all_scores[medianorder, :])
Plots.savefig(p_all_inds_heatmap, "$plots_output_dir/heatmap-mergedpop-$(job_name).png")

### top generalist plots
plot_lons = dataset_allpoints[INDEX_lon][1:eval_n_skip_points:end]
plot_lats = dataset_allpoints[INDEX_lat][1:eval_n_skip_points:end]

plot_scores = all_scores[medianorder[1], :]

plotorder = sortperm(plot_scores)

maps_plot(plot_lons[plotorder], plot_lats[plotorder], plot_scores[plotorder]; suptitle="$(short_job_name)-topmean-index$(medianorder[1])", score=round(nanmean(plot_scores), digits=2), pdf_path=plots_output_dir, marker_size=3, size=(1080, 640), vmin=0.0, vmax=1.0, extension="png")
chromo_draw(all_inds[medianorder[1]], "$plots_output_dir/generalist-1-graph.pdf")

########################################## TODO HERE
### pointwise-max models
print("Pointwise-max")
pointwise_max = true
if pointwise_max
    n_skip_sobol = 1
    global max_scores, max_score_modelindices
    max_scores = [maximum(all_scores[:, i]) for i in 1:size(all_scores, 2)]
    max_score_modelindices = [argmax(all_scores[:, i]) for i in 1:size(all_scores, 2)]
    # sobol_firstorder_indices = zeros(length(max_score_modelindices), 3)
    # sobol_totalorder_indices = zeros(length(max_score_modelindices), 3)
    # sobol_firstorder_confs = zeros(length(max_score_modelindices), 3)
    # sobol_totalorder_confs = zeros(length(max_score_modelindices), 3)

    # dominant_driver_indices = zeros(length(max_score_modelindices))
    # dominant_driver_confs = zeros(length(max_score_modelindices))

    plot_lons = dataset_allpoints[INDEX_lon][1:eval_n_skip_points:end]
    plot_lats = dataset_allpoints[INDEX_lat][1:eval_n_skip_points:end]
    plotorder = sortperm(max_scores)

    maps_plot(plot_lons[plotorder], plot_lats[plotorder], max_scores[plotorder]; suptitle="$(short_job_name)-pointwisemax", score=round(nanmean(max_scores), digits=2), pdf_path=plots_output_dir, marker_size=1.5, size=(1080, 640), vmin=0.0, vmax=1.0, extension="png")

    ### Sobol analysis of selected models 
    selected_models_sens_analyses = Dict()
    for model_i in unique(max_score_modelindices)
        println(model_i)
        data = SobolData(
            params = OrderedDict(:x1_E => Uniform(nanminimum(dataset_allpoints[INDEX_E]), nanmaximum(dataset_allpoints[INDEX_E])),
                    :x2_SLA => Uniform(nanminimum(dataset_allpoints[INDEX_SLA]), nanmaximum(dataset_allpoints[INDEX_SLA])),
                    :x3_FW => Uniform(nanminimum(dataset_allpoints[INDEX_FW]), nanmaximum(dataset_allpoints[INDEX_FW]))),
                N = 5_000
            )
        samples = GlobalSensitivityAnalysis.sample(data)
        samples = [samples[:,i] for i in 1:size(samples,2)]
        
        ind = all_inds[model_i]        
        CartesianGeneticProgramming.reset!(ind)
        Y = process(ind, samples)[1]
        analysis = analyze(data, Y; progress_meter=false)
        selected_models_sens_analyses[model_i] = analysis
    end

    dominant_driver_indices = zeros(length(max_score_modelindices))
    dominant_driver_confs = zeros(length(max_score_modelindices))

    ### find driver per point
    for point_i in 1:n_skip_sobol:lastindex(dataset_allpoints[INDEX_lon])
        println("$(point_i)/$(length(dataset_allpoints[INDEX_lon]))")
        analysis = selected_models_sens_analyses[max_score_modelindices[point_i]]
        
        first_order = analysis[:firstorder]

        dom_i = argmax(first_order)
        dominant_driver_indices[point_i] = dom_i
        dominant_driver_confs[point_i] = analysis[:firstorder_conf][dom_i]
    end

    plot_lons = dataset_allpoints[INDEX_lon][1:n_skip_sobol:end]
    plot_lats = dataset_allpoints[INDEX_lat][1:n_skip_sobol:end]
    plot_scores = dominant_driver_confs[1:n_skip_sobol:lastindex(dataset_allpoints[INDEX_lon])]
    
    bvals = [0.5, 1.5, 2.5, 3.5]
    mycolors = [
        "rgb(255,   0,      0)", 
        "rgb(50,     255,    50)",
        "rgb(50,     50,      255)", 
        ]
    dcolorscale = discrete_colorscale(bvals, mycolors)
    tvals = [1.35, 2.0, 2.65]
    model_names = ["Wave", "Sea level", "Fresh water"]
    maps_plot_discrete(plot_lons, plot_lats, dominant_driver_indices, tvals, model_names; suptitle="sobol-firstorder-dom-driver-map-ALL", pdf_path=plots_output_dir, marker_size=1.5, size=(1080, 640), cbar_len=250, vmin=1, vmax=3, cmap=dcolorscale)

    p_bar = Plots.bar((x -> countmap(model_names[Int.(dominant_driver_indices)])[x]).(model_names), xticks=(1:3, model_names), ylabel="N points", color=[:blue, :green, :red], legend=false, size=(280, 200))
    Plots.savefig(p_bar, "$plots_output_dir/pointwise-max-driver-barplot.pdf")
end

### cluster-based models
println("Cluster lookup")
cluster_based_lookup = true

function run_kmeans(df; k=9)
    mat = Matrix(df)
    feats = names(df)

    R = kmeans(mat[:, 3:end]', k)
    cluster_names = ["c$i" for i in 1:9]
    @assert nclusters(R) == k # verify the number of clusters

    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    return a, c, M
end

if cluster_based_lookup
    df_GCC = CSV.read("data/df_GCC_reduced.csv", DataFrame)
    subdf_GCC = select(df_GCC, [:lat, :lon, :he, :ns, :lu_water, :tr_zone_width])
    
    rowswithmissing = map(eachrow(subdf_GCC)) do r
        any(ismissing, r)
    end

    dropmissing!(subdf_GCC)

    plot_lons = dataset_allpoints[INDEX_lon][.!rowswithmissing]
    plot_lats = dataset_allpoints[INDEX_lat][.!rowswithmissing]

    possible_ks = vcat(2 .^ collect(1:6), collect(2^7:64:6013), [size(subdf_GCC)[1]]) 

    models_per_k = []
    cluster_assignments_per_k = []
    model_assignments_per_k = []
    mean_composite_scores = zeros(size(possible_ks))
    
    clusters_outdir = "$(plots_output_dir)/clusters2"
    if !isdir(clusters_outdir)
        mkdir(clusters_outdir)
    end

    for i_k_clusters in 1:lastindex(possible_ks)
        k_clusters = possible_ks[i_k_clusters]
        println("k_clusters $(k_clusters)")
        a, c, M = run_kmeans(subdf_GCC; k=k_clusters)

        cluster_ids = collect(1:k_clusters)
        # maps_plot(plot_lons, plot_lats, a; suptitle="GCC_$(k_clusters)clusters_map", score=0.0, pdf_path=clusters_outdir, marker_size=1.1, size=(1280, 640), vmin=1, vmax=k_clusters, extension="pdf")

        # p_a_hist = Plots.histogram(a, xticks=1:k_clusters)
        # Plots.savefig(p_a_hist, "$clusters_outdir/GCC_$(k_clusters)clusters_assignments_hist.pdf")

        ### select 1 model per cluster
        # model_indices = zeros(k_clusters)
        composite_scores = similar(plot_lats)
        clusters_models = []
        for cid in cluster_ids
            indices = findall(x -> x==cid, a)
            ind_means = nanmean(all_scores[:, indices], dims=2)[:]
            max_idx = argmax(ind_means)
            composite_scores[indices] .= all_scores[max_idx, indices]
            push!(clusters_models, max_idx)
        end

        mean_composite_scores[i_k_clusters] = nanmean(composite_scores)
        push!(models_per_k, clusters_models)
        push!(cluster_assignments_per_k, a)
        push!(model_assignments_per_k, clusters_models[a])
        # plot_scores = composite_scores
        # plotorder = sortperm(plot_scores)
        # maps_plot(plot_lons[plotorder], plot_lats[plotorder], plot_scores[plotorder]; suptitle="GCC_$(k_clusters)clusters_scores_map", score=round(nanmean(plot_scores), digits=2), pdf_path=clusters_outdir, marker_size=1.1, size=(1280, 640), vmin=0.0, vmax=1.0, extension="pdf")

        # bvals = [0.5, 1.5, 2.5, 3.5]
        # mycolors = [
        #     "rgb(255,   0,      0)", 
        #     "rgb(50,     255,    50)",
        #     "rgb(50,     50,      255)", 
        #     ]
        # dcolorscale = discrete_colorscale(bvals, mycolors)
        # tvals = [1.35, 2.0, 2.65]
        # model_names = ["Wave", "Sea level", "Fresh water"]
        # maps_plot_discrete(plot_lons, plot_lats, dominant_driver_indices[models_per_k[i_k_clusters]], tvals, model_names; suptitle="sobol-firstorder-dom-drivermap-GCC_$(k_clusters)clusters_drivers_map", pdf_path=clusters_outdir, marker_size=1.5, size=(1080, 640), cbar_len=250, vmin=1, vmax=3, cmap=dcolorscale)
    end
end

i_best_clustering = argmax(mean_composite_scores)

# N_in_final = [sum([(unique(models_per_k[j])[i] in unique(models_per_k[i_best_clustering])) for i in 1:lastindex(unique(models_per_k[j]))]) for j in 1:lastindex(possible_ks)]
# p_k_score = Plots.scatter(possible_ks, mean_composite_scores, xlabel="Number of clusters (k)", ylabel="Mean correlation", yguidefontcolor=:red, color=:red, legend=false, markersize=1.5, markerstrokewidth=0, grid=true)
# Plots.scatter!(twinx(), possible_ks, length.(unique.(models_per_k)), ylabel="N models", yguidefontcolor=:blue, color=:lightblue, legend=false, size=(600, 200), markersize=1.5, ylimits=(minimum(length.(unique.(models_per_k))), maximum(length.(unique.(models_per_k)))), markerstrokewidth=0, grid=true)
# Plots.scatter!(twinx(), possible_ks, N_in_final, ylabel="", yguidefontcolor=:green, color=:blue, legend=false, size=(400, 200), markersize=1.5, ylimits=(minimum(length.(unique.(models_per_k))), maximum(length.(unique.(models_per_k)))), markerstrokewidth=0)
# Plots.savefig(p_k_score, "$clusters_outdir/GCC_k_scatters_twinx.pdf")

models_raw_assignment = models_per_k[i_best_clustering]
best_assignment_model_counts = countmap(models_raw_assignment)
models_counts = sort(collect(best_assignment_model_counts), by=x->x[2])
models = [m[1] for m in models_counts]
mcounts = [m[2] for m in models_counts]

reiterate_model_selection = true
if reiterate_model_selection
    possible_thresholds = [25.0] ## fixed to 25.0 after testing. Replace with a list of possible thresholds to redo test (e.g. collect(1.0:100))
    thresholds_medianscores = similar(possible_thresholds)
    models_per_thresh = []
    for i_thresh in 1:lastindex(possible_thresholds)
        model_idx_pool = models[mcounts .> possible_thresholds[i_thresh]]
        k_clusters = possible_ks[i_best_clustering]
        a = cluster_assignments_per_k[i_best_clustering]
        cluster_ids = collect(1:k_clusters)
        
        composite_scores2 = similar(plot_lats)
        clusters_models2 = []
        for cid in cluster_ids
            indices = findall(x -> x==cid, a)
            ind_means = nanmean(all_scores[model_idx_pool, indices], dims=2)[:]
            max_idx = model_idx_pool[argmax(ind_means)]
            
            composite_scores2[indices] .= all_scores[max_idx, indices]
            push!(clusters_models2, max_idx)
        end

        push!(models_per_thresh, clusters_models2)
        thresholds_medianscores[i_thresh] = nanmean(composite_scores2)
        p_thresh_modelhist = Plots.histogram(models_per_thresh[i_thresh]; bins=0:1:2000, xticks=0:200:2000, title="Threshold: $(possible_thresholds[i_thresh]). Mean correlation: $(round(thresholds_medianscores[i_thresh], digits=2)). N models: $(length(unique(models_per_thresh[i_thresh])))", legend=false, ylabel="N coastal points", xlabel="Models", y_minorticks=0:1:maximum(models_per_thresh[i_thresh]), size=(680, 320), left_margin=0.5Plots.cm, bottom_margin=0.5Plots.cm)

        Plots.savefig(p_thresh_modelhist, "$clusters_outdir/GCC_thresh$(possible_thresholds[i_thresh])_Nmodelhist.pdf")

        plot_scores = composite_scores2
        plotorder = sortperm(plot_scores)
        maps_plot(plot_lons[plotorder], plot_lats[plotorder], plot_scores[plotorder]; suptitle="GCC_thresh$(possible_thresholds[i_thresh])____nmodels$(length(unique(clusters_models2)))__scores_map", score=round(nanmean(plot_scores), digits=2), pdf_path=clusters_outdir, marker_size=1.1, size=(1280, 640), vmin=0.0, vmax=1.0, extension="pdf")

        bvals = [0.5, 1.5, 2.5, 3.5]
        mycolors = [
            "rgb(255,   0,      0)", 
            "rgb(50,     255,    50)",
            "rgb(50,     50,      255)", 
            ]
        dcolorscale = discrete_colorscale(bvals, mycolors)
        tvals = [1.35, 2.0, 2.65]
        model_names = ["Wave", "Sea level", "Fresh water"]
        maps_plot_discrete(plot_lons, plot_lats, dominant_driver_indices[clusters_models2], tvals, model_names; suptitle="sobol-firstorder-dom-drivermap-GCC_thresh$(possible_thresholds[i_thresh])", pdf_path=clusters_outdir, marker_size=1.5, size=(1080, 640), cbar_len=250, vmin=1, vmax=3, cmap=dcolorscale)
    end
end

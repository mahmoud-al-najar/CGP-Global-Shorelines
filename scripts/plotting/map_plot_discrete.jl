using PlotlyJS, DataFrames, CSV, Plots, LaTeXStrings, StatsPlots
function maps_plot_discrete(lon, lat, vals, tvals, ttext; metric_name=L"r", cmap="RdYlGn_4", size=(1080, 720), marker_size=5, suptitle="suptitle", pdf_path="", cbar_len=200, cbar_wid=20, score=NaN, fits=NaN, vmin=0.0, vmax=1.0, extension="pdf")
    marker = attr(size=marker_size,
        color=vals, 
        colorscale=cmap, 
        #title=attr(text=metric_name, side="right"), 
        colorbar=attr(tickvals=tvals, ticktext=ttext, len=cbar_len,  lenmode="pixels", thickness=cbar_wid, thicknessmode="pixels"), 
        reversescale=true)

    trace = scattergeo(; locationmode="ISO-3",
        lat=lat,
        lon=lon,
        color=vals,
        marker=marker,
        hoverinfo="text",
        text=vals,
        cauto=true)

    geo = attr(scope="world",
        projection_type="natural earth",
        showland=true,
        landcolor="rgb(217, 217, 217)",
        subunitwidth=1,
        countrywidth=1,
        subunitcolor="rgb(150, 150, 150)",
        countrycolor="rgb(150, 150, 150)",
        showcoastline=true,
        showrivers=true,
        showcountries=false,
        resolution=50,
        fitbounds="locations")

    if !isnan(score)
        figsuptitle = "$suptitle\tr=$score"
    else
        figsuptitle = suptitle
    end

    if typeof(fits) == Vector{Float64}
        figsuptitle = "$figsuptitle\nfitness=$fits"
    end

    layout = Layout(; title=figsuptitle, showlegend=false, geo=geo, width=size[1], height=size[2], dpi=600)
    p1 = PlotlyJS.plot(trace, layout)
    PlotlyJS.savefig(p1, "$pdf_path/satellite-$suptitle.$extension")
end

function discrete_colorscale(boundary_vals, mycolors)
    #boundary_vals - vector of values bounding intervals/ranges of interest
    #colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],1<=k < length(boundary_vals)
    #returns the plotly  discrete colorscale
    if length(boundary_vals) != length(mycolors)+1
           error("length of boundary values should be equal to  length(mycolors)+1")
    end
    bvals = sort(boundary_vals)     
    nvals = [(v-bvals[1])/(bvals[end]-bvals[1]) for v in bvals]  #normalized values
    
    dcolorscale = [] #discrete colorscale
    for k in 1:length(mycolors)
        append!(dcolorscale, [[nvals[k], mycolors[k]], [nvals[k+1], mycolors[k]]])
    end    
    return dcolorscale 
      
end

# function colorbar_ticks(bvals)
#     tickvals = [sum(bvals[k:k+1])/2 for k in 1:length(bvals)-1] #position with respect to bvals, to be placed ticktext 
#     ticktext = String[]
#     push!(ticktext, "<$(bvals[2])")
#     for k in 2:length(bvals)-2
#         push!(ticktext, "$(bvals[k])-$(bvals[k+1])")
#     end        
#     push!(ticktext, ">$(bvals[end-1])")  
#     return tickvals, ticktext
# end  

# include("../global_zones/extract_zones.jl")
# datasets_waves = load_WAVE_dominated_zones()
# wav_lat = datasets_waves[13]
# wav_lon = datasets_waves[14]
# wav_labels = ones(length(wav_lat)) .* 3

# datasets_river = load_RIVER_dominated_zones()
# riv_lat = datasets_river[13]
# riv_lon = datasets_river[14]
# riv_labels = ones(length(riv_lat))

# datasets_sla = load_SLA_dominated_zones()
# sla_lat = datasets_sla[13]
# sla_lon = datasets_sla[14]
# sla_labels = ones(length(sla_lat)) .* 2

# plot_lats = [riv_lat..., sla_lat..., wav_lat...]
# plot_lons = [riv_lon..., sla_lon..., wav_lon...]
# plot_labels = [riv_labels..., sla_labels..., wav_labels...]


# bvals = [0.5, 1.5, 2.5, 3.5]
# mycolors = [
#     "rgb(0, 102, 153)", 
#     "rgb(126, 186, 181)",
#     "rgb(255, 255, 179)", 
#     ]

# mycolors = [
#     "rgb(255,   50,      50)", 
#     "rgb(50,     255,    50)",
#     "rgb(50,     50,      255)", 
#     ]
# dcolorscale = discrete_colorscale(bvals, mycolors)

# tvals = [1.35, 2, 2.65]
# ttext = ["River", "SLA", "Wave"]

# maps_plot_discrete(plot_lons, plot_lats, plot_labels, tvals, ttext; suptitle="discrete", pdf_path="graphing/", marker_size=5, size=(1080, 640), cbar_len=250, vmin=1, vmax=3, cmap=dcolorscale)


# ##############################################3
# bvals = [0.5, 1.5, 2.5]
# mycolors = [
#     # "rgb(0, 102, 153)", 
#     "rgb(126, 186, 181)",
#     "rgb(255, 255, 179)", 
#     ]

# mycolors = [
#     # "rgb(255,   50,      50)", 
#     "rgb(50,     255,    50)",
#     "rgb(50,     50,      255)", 
#     ]
# dcolorscale = discrete_colorscale(bvals, mycolors)

# # tvals = [1.35, 2, 2.65]
# # ttext = ["River", "SLA", "Wave"]
# tvals = [1.25, 1.75]
# ttext = ["Global-SLA", "Global-P"]

# maps_plot_discrete(plot_lons, plot_lats, pairwise_models, tvals, ttext; suptitle="discrete", pdf_path="graphing/", marker_size=3, size=(1080, 640), cbar_len=250, vmin=1, vmax=3, cmap=dcolorscale)

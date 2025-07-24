using PlotlyJS, DataFrames, CSV, Plots, LaTeXStrings, StatsPlots
function maps_plot(lon, lat, vals; metric_name=L"r", cmap="RdYlGn_4", size=(2080, 1200), marker_size=5, suptitle="suptitle", pdf_path="", cbar_len=200, cbar_wid=20, score=NaN, fits=NaN, vmin=0.0, vmax=1.0, histogram=false, projection_type="natural earth", extension="pdf")
    marker = attr(size=marker_size,
        color=vals, 
        colorscale=cmap, 
        cmin=vmin,
        cmax=vmax,
        colorbar=attr(title=attr(text=metric_name, side="right"), len=cbar_len, lenmode="pixels", thickness=cbar_wid, thicknessmode="pixels"), reversescale=true)

    trace = scattergeo(; locationmode="ISO-3",
        lat=lat,
        lon=lon,
        color=vals,
        marker=marker,
        hoverinfo="text",
        text=vals,
        cauto=true)

    geo = attr(scope="world",
        projection_type=projection_type,
        projection_scale=1,
        showland=true,
        landcolor="rgb(217, 217, 217)",
        # landcolor="rgb(190, 190, 190)",
        subunitwidth=1,
        countrywidth=1,
        # subunitcolor="rgb(255,255,255)",
        # countrycolor="rgb(255,255,255)",
        subunitcolor="rgb(150, 150, 150)",
        countrycolor="rgb(150, 150, 150)",
        showcoastline=true,
        showrivers=true,
        showcountries=false,
        resolution=50,
        fitbounds="locations")

    if !isnan(score)
        figsuptitle = "$suptitle\n$metric_name=$score"
    else
        figsuptitle = suptitle
    end

    if typeof(fits) == Vector{Float64}
        figsuptitle = "$figsuptitle\nfitness=$fits"
    end

    layout = Layout(; title=figsuptitle, showlegend=false, geo=geo, width=size[1], height=size[2], dpi=600)
    p1 = PlotlyJS.plot(trace, layout)
    PlotlyJS.savefig(p1, "$pdf_path/satellite-$suptitle.$extension")
    
    if histogram
        p_hist = StatsPlots.histogram(vals, nbins=10)
        StatsPlots.savefig(p_hist, "$pdf_path/histogram-$suptitle.$extension")
    end
end


# include("../global_zones/extract_zones.jl")
# zones = load_all_zones(;normalize=true)
# lons = []
# lats = []

# for zone in zones
#     global lons = vcat(lons, zone[14])
#     global lats = vcat(lats, zone[13])
# end

# projections = ["mercator", "natural earth", "satellite"]
# for proj in projections
#     maps_plot(lons, lats, ones(length(lons)); pdf_path="/home/mn/JuliaProjects/CartesianGeneticProgramming.jl/global_zones/projections", suptitle=replace(proj, " "=> ""), marker_size=4, cmap="Bluered", projection_type=proj)
# end

using MAT
using SeisNoise
using Metrics
using Statistics
using NaNStatistics
using Plots
using Dates
using Dierckx


const MATLAB_EPOCH = Dates.DateTime(-1,12,31)
date2num(d::Dates.DateTime) = Dates.value(d-MATLAB_EPOCH)/(1000*60*60*24)
num2date(n::Number) =  MATLAB_EPOCH + Dates.Millisecond(round(Int64, n*1000*60*60*24))

function minmax_normalize(x::Array{Float64}; vmin=0.0, vmax=1.0)
    return (vmax-vmin) .* (x .-  nanminimum(x)) ./ (nanmaximum(x) - nanminimum(x)) .+ vmin
end

function minmax_normalize_globalmaxmin(x::Array{Float64}, globalvmin, globalvmax; vmin=0.0, vmax=1.0)
    return (vmax-vmin) .* (x .-  globalvmin) ./ (globalvmax - globalvmin) .+ vmin
end


## indices used in dataset array
const INDEX_t = 1
const INDEX_X = 2
const INDEX_E = 3
const INDEX_SLA = 4
const INDEX_FW = 5
const INDEX_lat = 6
const INDEX_lon = 7

const END_I = 200 # length of training set
const datadir = "data"
const dataset_file = "$(datadir)/processed_dataset.mat"

interp1(x,v,xq) = Spline1D(x, v; k=1)(xq)

function fall_velocity(D, Tw)
    # D = Grain size [m]
    # Tw = Temp in degrees [C]
    # w returned in m/s
    D=D*100

    ROWs=2.75	# Density of sand (Mark 2.75, Kristen, 2.65?)
    g=981		# Gravity cm/s^2

    T   =[5, 10, 15, 20, 25]
    v   =[0.0157, 0.0135, 0.0119, 0.0105, 0.0095]
    ROW =[1.028, 1.027, 1.026, 1.025, 1.024] 

    vw=interp1(T,v,Tw)
    ROWw=interp1(T,ROW,Tw)

    A = ((ROWs-ROWw)*g*(D.^3))./(ROWw*(vw.^2))

    if A < 39
        w=((ROWs-ROWw)*g*(D.^2))./(18*ROWw*vw)
    else
        if A < 10^4   
            w=((((ROWs-ROWw)*g./ROWw).^0.7)*(D.^1.1))./(6*(vw.^0.4))
        else
            w=sqrt(((ROWs-ROWw)*g*D)./(0.91*ROWw))
        end
    end

    w=w./100 # convert to SI (m/s)
    return w
end

function calcPb(H)  # ,T)
    #function to calculate wave power at breaking
    #P = ECn, where E = wave Energy, Cn=Cg = group velocity

    g=9.81;
    rho = 1025;
    gamma=0.78;
    E = 1 ./ 16 .* rho .* g .* H.^2;
    Cg=sqrt.(g.*H./gamma);
    P=E.*Cg;
    return P
end

lowF = T -> 1.5 * T
highF = T -> 0.75 * T
passband = (S,lowf,highf) -> movmean(S - movmean(S, lowf), highf)

function smooth(lower, upper, x)
    window_lengths = collect(lower:upper)
    ma_x = Array{Float64}(undef, length(window_lengths), length(x))
    for i in 1:lastindex(window_lengths)
        l = window_lengths[i]
        ma_x[i, :] = passband(x, lowF(l), highF(l)) ./ l
    end
    
    return sum(ma_x, dims=1)[:]    
end

function global_dataset(;normalize=false, calibration=false, filter_limits=(NaN, NaN))
    if !isfile(dataset_file)
        ds_X = matread("$(datadir)/Shorelines_global_20231101_shift.mat")
        ds_drivers = matread("$(datadir)/shoreline_forcing_20231103.mat")
        ds_daily_drivers = matread("$(datadir)/Wave_dailyAVG_AlongshoreGrid_1993_2023.mat")
    
        ### last 5 timesteps in River_drivers are always NaN, adding -5 to below
        t = ds_X["time"][109:109+323-5]
        X=detrend(ds_X["X_safe"][:, 109:109+323-5]')
    
        River_drivers = ds_drivers["rivdis2"][1:size(X, 1), 1:size(X, 2)]
        SLA_drivers = ds_drivers["sla2"][1:size(X, 1), 1:size(X, 2)]
        EWave_drivers = ds_drivers["ewave2"][1:size(X, 1), 1:size(X, 2)]
    
        lat = ds_drivers["LAT_loc"][1:size(X, 2)]
        lon = ds_drivers["LON_loc"][1:size(X, 2)]
    
        ### filter 325 first points in X, all NaN
        idx_notnan = [count(isnan.(X[:, i])) != size(X, 1) for i in 1:size(X, 2)]
        idx_nan = [count(isnan.(X[:, i]))    == size(X, 1) for i in 1:size(X, 2)]
    
        ## uncomment to plot map of nan points
        # maps_plot(lon[idx_nan], lat[idx_nan], ones(length(idx_nan)); suptitle="nan_points", score=count(idx_nan), pdf_path="$datadir/", marker_size=3, size=(1080, 640), metric_name="nan points:", vmin=1.0, vmax=1.0)
    
        X = X[:, idx_notnan]
        lat = lat[idx_notnan]
        lon = lon[idx_notnan]
    
        River_drivers = River_drivers[:, idx_notnan]
        SLA_drivers = SLA_drivers[:, idx_notnan]
        EWave_drivers = EWave_drivers[:, idx_notnan]
    
        ### daily wave conditions
        ## filtering spatial dim: points (lat-lon) are ordered in raw files
        daily_dates = ds_daily_drivers["TIME"]
        daily_datetimes = [DateTime(daily_dates[x, 1], daily_dates[x, 2], daily_dates[x, 3], daily_dates[x, 4]) for x in 1:size(daily_dates, 1)]
        daily_Hs = ds_daily_drivers["Hs"][:, 1:length(ds_X["lonX"])][:, idx_notnan]
        daily_Tp = ds_daily_drivers["Tp"][:, 1:length(ds_X["lonX"])][:, idx_notnan]
        daily_Dir = ds_daily_drivers["Dir"][:, 1:length(ds_X["lonX"])][:, idx_notnan]
    
        ### monthly means
        monthly_datetimes = num2date.(t)
    
        monthly_Hs = similar(EWave_drivers)
        monthly_Tp = similar(EWave_drivers)
        monthly_Dir = similar(EWave_drivers)
    
        for xi in 1:length(lon)
            for i_datetime in 1:length(monthly_datetimes)
                datetime = monthly_datetimes[i_datetime]
                year = Dates.year(datetime)
                month = Dates.month(datetime)
                
                month_hs = []
                month_tp = []
                month_dir = []
    
                for i_daily_t in 1:size(daily_dates, 1)
                    d_year = daily_dates[i_daily_t, 1]
                    d_month = daily_dates[i_daily_t, 2]
                    if d_year == year && d_month == month
                        push!(month_hs, daily_Hs[i_daily_t, xi])
                        push!(month_tp, daily_Tp[i_daily_t, xi])
                        push!(month_dir, daily_Dir[i_daily_t, xi])
                    end
                end
                
                monthly_Hs[i_datetime, xi] = mean(month_hs)
                monthly_Tp[i_datetime, xi] = mean(month_tp)
                monthly_Dir[i_datetime, xi] = mean(month_dir)
                # println("$(month)/$(year) -- $(length(month_hs)) days")
            end
            println("$(xi)/$(length(lon))")
        end
    
        ### removing anomalies in wave data. 
        empties = [count(x -> x==0.030884085f0, monthly_Hs[:, i]) for i in 1:size(monthly_Hs, 2)]
        to_keep = empties .== 0
        X = X[:, to_keep]
        EWave_drivers = EWave_drivers[:, to_keep]
        SLA_drivers = SLA_drivers[:, to_keep]
        River_drivers = River_drivers[:, to_keep]
        monthly_Hs = monthly_Hs[:, to_keep]
        monthly_Tp = monthly_Tp[:, to_keep]
        monthly_Dir = monthly_Dir[:, to_keep]
        lat = lat[to_keep]
        lon = lon[to_keep]

        monthly_P = similar(monthly_Hs)
        monthly_Omega = similar(monthly_Hs)

        d50=0.25
        Tw=15
        w = fall_velocity(d50/1000, Tw)

        for i in 1:lastindex(lon)
            monthly_P[:, i] = calcPb(monthly_Hs[:, i]) .^ 0.5
            monthly_Omega[:, i] = monthly_Hs[:, i] ./ (monthly_Tp[:, i] * w)
        end

        # save dataset
        ds_file = matopen(dataset_file, "w")
        write(ds_file, "t", t)
        write(ds_file, "X", X)
        write(ds_file, "EWave_drivers", EWave_drivers)
        write(ds_file, "SLA_drivers", SLA_drivers)
        write(ds_file, "River_drivers", River_drivers)
        write(ds_file, "monthly_Hs", monthly_Hs)
        write(ds_file, "monthly_Tp", monthly_Tp)
        write(ds_file, "monthly_Dir", monthly_Dir)
        write(ds_file, "lat", lat)
        write(ds_file, "lon", lon)

        write(ds_file, "monthly_P", monthly_P)
        write(ds_file, "monthly_Omega", monthly_Omega)

        close(ds_file)
    end
    
    ds = matread(dataset_file)
    t = ds["t"]
    X = Matrix{Float64}(ds["X"])
    EWave_drivers = Matrix{Float64}(ds["EWave_drivers"])
    SLA_drivers = Matrix{Float64}(ds["SLA_drivers"])
    River_drivers = Matrix{Float64}(ds["River_drivers"])
    monthly_Hs = Matrix{Float64}(ds["monthly_Hs"])
    monthly_Tp = Matrix{Float64}(ds["monthly_Tp"])
    monthly_Dir = Matrix{Float64}(ds["monthly_Dir"])
    monthly_P = Matrix{Float64}(ds["monthly_P"])
    monthly_Omega = Matrix{Float64}(ds["monthly_Omega"])
    lat = ds["lat"]
    lon = ds["lon"]

    if !isnan(filter_limits[1])
        lower = filter_limits[1]
        upper = filter_limits[2]

        for i in 1:lastindex(lon)
            X[:, i] = smooth(lower, upper, X[:, i])
            EWave_drivers[:, i] = smooth(lower, upper, EWave_drivers[:, i])
            SLA_drivers[:, i] = smooth(lower, upper, SLA_drivers[:, i])
            River_drivers[:, i] = smooth(lower, upper, River_drivers[:, i])
            monthly_Hs[:, i] = smooth(lower, upper, monthly_Hs[:, i])
            monthly_Tp[:, i] = smooth(lower, upper, monthly_Tp[:, i])
            monthly_P[:, i] = smooth(lower, upper, monthly_P[:, i])
            monthly_Omega[:, i] = smooth(lower, upper, monthly_Omega[:, i])
        end
    end

    if normalize
        for i in 1:lastindex(lon)
            X[:, i] = minmax_normalize(X[:, i]; vmin=-1.0, vmax=1.0)
            EWave_drivers[:, i] = minmax_normalize(EWave_drivers[:, i]; vmin=0.0, vmax=1.0)
            SLA_drivers[:, i] = minmax_normalize(SLA_drivers[:, i]; vmin=0.0, vmax=1.0)
            River_drivers[:, i] = minmax_normalize(River_drivers[:, i]; vmin=0.0, vmax=1.0)
            monthly_Hs[:, i] = minmax_normalize(monthly_Hs[:, i]; vmin=0.0, vmax=1.0)
            monthly_Tp[:, i] = minmax_normalize(monthly_Tp[:, i]; vmin=0.0, vmax=1.0)
            # monthly_Dir[:, i] = minmax_normalize(monthly_Dir[:, i]; vmin=0.0, vmax=1.0)
            monthly_P[:, i] = minmax_normalize(monthly_P[:, i]; vmin=0.0, vmax=1.0)
            monthly_Omega[:, i] = minmax_normalize(monthly_Omega[:, i]; vmin=0.0, vmax=1.0)
        end
    end

    if calibration
        t = t[1:END_I]
        X = X[1:END_I, :]
        EWave_drivers = EWave_drivers[1:END_I, :]
        SLA_drivers = SLA_drivers[1:END_I, :]
        River_drivers = River_drivers[1:END_I, :]
        monthly_Hs = monthly_Hs[1:END_I, :]
        monthly_Tp = monthly_Tp[1:END_I, :]
        monthly_Dir = monthly_Dir[1:END_I, :]
        monthly_P = monthly_P[1:END_I, :]
        monthly_Omega = monthly_Omega[1:END_I, :]
    end

    # removing anomalies from post-hoc monthly wave params (Tp)
    to_keep = findall(x-> x==0, [count(isnan.(monthly_Tp[:, i])) for i in 1:lastindex(lon)])
    X = X[:, to_keep]
    EWave_drivers = EWave_drivers[:, to_keep]
    SLA_drivers = SLA_drivers[:, to_keep]
    River_drivers = River_drivers[:, to_keep]
    monthly_Hs = monthly_Hs[:, to_keep]
    monthly_Tp = monthly_Tp[:, to_keep]
    monthly_Dir = monthly_Dir[:, to_keep]
    lat = lat[to_keep]
    lon = lon[to_keep]
    
    dataset_allpoints = [t, X, EWave_drivers, SLA_drivers, River_drivers, lat, lon]
    dataset_allpoints
end

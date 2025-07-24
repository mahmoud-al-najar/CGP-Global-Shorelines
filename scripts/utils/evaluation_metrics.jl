using Statistics
using Metrics


function rms(o, m; digits=5)
    return round(sqrt(mean((o - m).^2)), digits=digits)
end

function corr(o, m; digits=5)
    return round(cor(o, m), digits=digits)
end

function mielke(o, m; digits=5)
    N = length(o)
    # λ = 1 - (N^-1 * sum((o .- m).^2)) / (std(o)^2 + std(m)^2 + (mean(o) - mean(m))^2)
    λ = 1 - (N^-1 * sum((o .- m).^2)) / (var(o) + var(m) + (mean(o) - mean(m))^2)
    return round(λ, digits=digits)
end

function concordance_correlation_coefficient(y_true, y_pred; digits=5)
    correlation = cor(y_true,y_pred)
    
    mean_true = mean(y_true)
    mean_pred = mean(y_pred)
    
    var_true = var(y_true)
    var_pred = var(y_pred)
    
    sd_true = std(y_true)
    sd_pred = std(y_pred)
    
    numerator = 2 * correlation * sd_true * sd_pred
    
    denominator = var_true + var_pred + (mean_true - mean_pred)^2

    return round(numerator/denominator, digits=digits)
end

function calc_r2(o, m; digits=5)
    round(r2_score(o, m), digits=digits)
end

function skill(o, m; digits=5)
    # num = sum((o .- m).^2)
    num = rms(o, m)
    score = 1.0 - (num / var(o))
    return round(score, digits=digits)
end

function agreement(o, m; digits=5)
    num = sum((o .- m).^2)
    m_res = abs.(m .- mean(o))
    o_res = abs.(o .- mean(o))
    den = sum((m_res .+ o_res).^2)
    score = 1.0 - (num / den)
    return round(score, digits=digits)
end

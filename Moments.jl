module Moments
export m

using SymPy
using HCubature
using Distributions
using LinearAlgebra
using SpecialFunctions
using Memoization
using Combinatorics

import Base: +, *

function sigma(l, k)
    sigma_l = factorial(l) / doublefactorial(2l + 1)
    lambda_lk = factorial(l + k + 1/2) / (factorial(big(k)) * factorial(1/2))
    return Float64(sigma_l * lambda_lk)
end

function c(l, k, m)
    num = (-1)^m * factorial(k + l + 1/2)
    denom = factorial(big(m)) * factorial(big(k - m)) * factorial(l + m + 1/2)
    Float64(num / denom)
end

function L(l, k)
    return x -> sum([c(l, k, m) * x^m for m in 0:k])
end

function p_sym(l::Int64)
    vx, vy, vz = symbols("vx vy vz")
    v = [vx, vy, vz]
    if l == 0
        return [Sym(1)]
    elseif l == 1
        return v
    else
        lower_order = p_sym(l - 1)
        speed_squared = vx^2 + vy^2 + vz^2
        lcat = (l) -> (m1, m2) -> cat(m1, m2; dims=l)
        outer_product = reduce(lcat(l), [vi .* lower_order for vi in v])
        jac = reduce(lcat(l), [diff.(lower_order, vi) for vi in v])
        return outer_product .- (jac .* (speed_squared / (2l - 1)))
    end
end

function p(l::Int64)
    vx, vy, vz = symbols("vx vy vz")
    v = [vx, vy, vz]
    varargsify = p_l -> function(v)
        return p_l(v...)
    end
    varargsify.(lambdify.(p_sym(l), (v,))) # Prevents broadcast over vars
end

function p(l, k)
    p_l = p(l)
    L_lk = L(l, k)
    combine = p -> function(v)
        c = norm(v)
        return p(v) * L_lk(c^2)
    end
    combine.(p_l)
end

function integrand_func(p, sigma, f)
    return function(v)
        return p(v) * f(v) / sqrt(sigma)
    end
end

function m(l, k, f, a, b)
    p_lk = p(l, k)
    s = sigma(l, k)
    m = function(p_expr)
        integrand = integrand_func(p_expr, s, f)
        return hcubature(integrand, a, b, atol=0.01)[1]
    end
    m.(p_lk)
end

function f_lk_hat(l, k, m_lk)
    p_lk = p(l, k)
    sigma_lk = sigma(l, k)
    f_M = maxwellian([0., 0., 0.])
    fuse = v -> (p -> p(v))
    fused = v -> fuse(v).(p_lk)
    v -> sum(fused(v) .* m_lk * f_M(v) / sqrt(sigma_lk))
end

function moment_hierarchy(f0, lmax, kmax)
    moment_hierarchy::Array{TensorMomentVector, 1} = []
    for l in 0:lmax
        moments::Array{TensorMoment{l}} = []
        for k in 0:kmax
            mlk = m(l, k, f0, a, b)
            if l == 0
                mlk = fill(mlk[1], ()) # Convert the singleton rank-1 array to a rank-0 array
            end
            push!(moments, TensorMoment(mlk))
        end
        push!(moment_hierarchy, TensorMomentVector(moments))
    end
    MomentHierarchy(moment_hierarchy)
end

function moment_reconstruction(f0, lmax, kmax)
    a = [-10., -10., -10.]
    b = [10., 10., 10.]
    hats = []
    for l in 0:lmax
        for k in 0:kmax
            mlk = m(l, k, f0, a, b)
            display(mlk)
            flk_hat = f_lk_hat(l, k, mlk)
            push!(hats, flk_hat)
        end
    end
    return v -> sum([hat(v) for hat in hats])
end

function maxwellian(V)
    return function(v)
        c = norm(V - v)
        return pi^(-3/2) * exp(-c^2)
    end
end

fM = maxwellian([0., 0., 0.])
f0 = fM

a = [-10., -10., -10.]
b = [10., 10., 10.]

# A vector of tensors of rank L. The kth component contains the order (l, k) tensor moment.
mutable struct TensorMoment{L}
    tensor::Array{Float64, L}
end

function (*)(moment::TensorMoment{L}, r::Number)::TensorMoment{L} where {L}
    TensorMoment(moment.tensor * r)
end

mutable struct TensorMomentVector{L, K}
    moments::Array{TensorMoment{L}, 1}
    TensorMomentVector(moments::Array{TensorMoment{L}, 1}, k) where {L} = size(moments) == (K,) ? new{K, L}(moments) : error("Expected array of dimension " + K)
end

function (*)(moment_vec::TensorMomentVector{L, K}, r::Number)::TensorMomentVector{L, K} where {L, K}
    return TensorMomentVector{L, K}(moment_vec.moments .* r)
end

function (*)(moment_vec::TensorMomentVector{L, K}, r::AbstractVector{Float64})::TensorMomentVector{L, K} where {L, K}
    return TensorMomentVector{L, K}(moment_vec.moments .* r)
end

function (+)(a::TensorMomentVector{L, K}, b::TensorMomentVector{L, K})::TensorMomentVector{L, K} where {L, K}
    return TensorMomentVector{L, K}(a.moments + b.moments)
end

function (*)(moment_vec::TensorMomentVector{L, K}, A::AbstractMatrix{Float64})::TensorMomentVector{L, K} where {L, K}
    return sum(moment_vec * A[j, :] for j in size(A)[2])
end

# The hierarchy of tensor moments.
mutable struct MomentHierarchy{L, K}
    hierarchy::Array{TensorMomentVector{Int64, K}, 1}
end

function (*)(moment_hierarchy::MomentHierarchy{L, K}, matrices::Vector{AbstractMatrix{Float64}}) where {L, K}
    MomentHierarchy([
        moment_hierarchy.hierarchy[l] * matrices[l] for l in 1:(L+1)
    ])
end

end

module CollisionOperator

using ..Moments
using Memoization

function a_E(l, m, n)
    if n == 1
        return  -(l + 2*m) - (1/2)*(l + 1)*l;
    elseif n == 2
        return 2*m*(l + m - 1) + (3/4)*(l - 1)*l;
    else
        error("Invalid value of n")
    end
end

function a_e(l, m, n)
    if n == 1
        return 2*(l + 2*m);
    elseif n == 2
        return -(2*m*(l + m - 1)) - (3/4)*(l - 1)*l;
    else
        error("Invalid value of n")
    end
end

function b_e(l, m, n)
    if n == 0
        return 2.;
    elseif n == 1
        return -((4*(l^2 + l - 1))/((2*l - 1)*(2*l + 3)));
    else
        error("Invalid value of n")
    end
end

function b_sub_sup(l, m, sub, sup)
    if sub == 1
        if sup == 0
            return (2*(l + 1)*(l + 2)*(2*l + 2*m + 3))/((2*l + 1)*(2*l + 3)) - 4/(2*l + 1);
        elseif sup == 1
            return -((4*l*(l - 1))/((2*l - 1)*(2*l + 1)));
        else
            error("Invalid superscript for b")
        end
    elseif sub == -1
        if sup == 0
            return -((4*(l - 1)*l*(m + 1))/((2*l - 1)*(2*l + 1))) - 4/(2*l + 1);
        elseif sup == 1
            return (4*(l + 1)*(l + 2))/ ((2*l + 1)*(2*l + 3));
        else
            error("Invalid superscript for b")
        end
    else
        error("Invalid subscript for b")
    end
end

function E(k, i)
    return (1/2)*sum([(factorial(k)*e(i + j))/factorial(j) for j in 0:k]);
end

function e(j)
    return ((1/2)^(j + 1/2)*factorial(j - 1/2))/factorial(-(1/2));
end

function A(l, q, m)
    return sum(e(l + m - n + q + 1) * a_e(l, m, n) + E(l + m - n + q, 0) * a_E(l, m, n) for n in 1:2)
end

function B(l, q, m)
    return sum(e(l + m + n + q + 1) * b_e(l, m, n) + E(m, l + n + q + 1) * b_sub_sup(l, m, -1, n) + E(n + q, l + m + 1) * b_sub_sup(l, m, 1, n) for n in 0:1)
end

function C(l, p, k)
    sum(Moments.c(l, p, q) * Moments.c(l, k, m) * (A(l, q, m) + B(l, q, m)) for m=0:k, q=0:p)
end

@memoize Dict function collision_matrix(l, K)
    [C(l, p, k) for p=0:K, k=0:K]
end


end

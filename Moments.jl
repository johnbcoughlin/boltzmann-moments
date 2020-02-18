module Moments
export m

using SymPy
using HCubature
using Distributions
using LinearAlgebra
using SpecialFunctions
using Memoization
using Combinatorics

import Base: +, *, show

@memoize Dict function sigma(l, k)
    sigma_l = factorial(big(l)) / doublefactorial(2l + 1)
    lambda_lk = SpecialFunctions.gamma(l + k + 3/2) / (factorial(big(k)) * factorial(1/2))
    return Float64(sigma_l * lambda_lk)
end

@memoize Dict function c(l, k, m)
    num = (-1)^m * SpecialFunctions.gamma(l + k + 3/2)
    denom = factorial(big(k - m)) * SpecialFunctions.gamma(l + m + 3/2) * factorial(big(m))
    Float64(num / denom)
end

function L(l, k)
    return x -> sum([c(l, k, m) * x^m for m in 0:k])
end

@memoize Dict function p_sym(l::Int64)
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

@memoize Dict function p(l::Int64)
    vx, vy, vz = symbols("vx vy vz")
    v = [vx, vy, vz]
    varargsify = p_l -> function(v)
        return p_l(v...)
    end
    varargsify.(lambdify.(p_sym(l), (v,))) # Prevents broadcast over vars
end

@memoize Dict function p(l, k)
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

# A vector of tensors of rank L. The kth component contains the order (l, k) tensor moment.
mutable struct TensorMoment{L}
    tensor::Array{Float64, L}
end

@memoize function m(l, k, f, a, b)
    p_lk = p(l, k)
    s = sigma(l, k)
    m = function(p_expr)
        integrand = integrand_func(p_expr, s, f)
        return hcubature(integrand, a, b, atol=0.001)[1]
    end
    m.(p_lk)
end

function f_lk_hat(l, k, m_lk::TensorMoment{L}, fM) where {L}
    p_lk = p(l, k)
    sigma_lk = sigma(l, k)
    fuse = v -> (p -> p(v))
    fused = v -> fuse(v).(p_lk)
    v -> sum(fused(v) .* m_lk.tensor * fM(v) / sqrt(sigma_lk))
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
        push!(moment_hierarchy, TensorMomentVector{kmax, l}(moments))
    end
    MomentHierarchy{kmax, lmax}(moment_hierarchy)
end

function maxwellian(V, n, v_T)
    return function(v)
        c = norm(V - v)
        return n * pi^(-3/2) * v_T^(-3) * exp(-c^2)
    end
end

fM = maxwellian([0., 0., 0.], 1., 1.)
f0 = fM

a = [-10., -10., -10.]
b = [10., 10., 10.]

function (*)(moment::TensorMoment{L}, r::Number)::TensorMoment{L} where {L}
    TensorMoment(moment.tensor * r)
end

mutable struct TensorMomentVector{K, L}
    moments::Array{TensorMoment{L}, 1}
    TensorMomentVector{K, L}(moments::Array{TensorMoment{L}, 1}) where {K, L} = size(moments) == (K+1,) ? new{K, L}(moments) : error("Expected array of dimension $K")
end

function (*)(moment_vec::TensorMomentVector{K, L}, r::Number)::TensorMomentVector{K, L} where {K, L}
    return TensorMomentVector{K, L}(moment_vec.moments .* r)
end

function (*)(r::AbstractVector{Float64}, moment_vec::TensorMomentVector{K, L})::TensorMomentVector{K, L} where {K, L}
    return TensorMomentVector{K, L}(moment_vec.moments .* r)
end

function (+)(a::TensorMomentVector{K, L}, b::TensorMomentVector{K, L})::TensorMomentVector{K, L} where {K, L}
    return TensorMomentVector{K, L}(a.moments + b.moments)
end

function (*)(A::AbstractMatrix{Float64}, moment_vec::TensorMomentVector{K, L})::TensorMomentVector{K, L} where {K, L}
    return sum(moment_vec * A[j, :] for j in size(A)[2])
end

Base.length(vec::TensorMomentVector{K, L}) where {K, L} = length(vec.moments)

Base.show(io::IO, moment_vec::TensorMomentVector{K, L}) where {K, L} = show(io, moment_vec.moments)

# The hierarchy of tensor moments.
mutable struct MomentHierarchy{KMax, LMax}
    hierarchy::Array{TensorMomentVector{KMax}, 1}
end

function (*)(moment_hierarchy::MomentHierarchy{K, L}, matrices::Array{Array{Float64, 2}, 1}) where {K, L}
    MomentHierarchy{K, L}([
        moment_hierarchy.hierarchy[l] * matrices[l] for l in 1:(L+1)
    ])
end

function (+)(a::MomentHierarchy{K, L}, b::MomentHierarchy{K, L})::MomentHierarchy{K, L} where {K, L}
    MomentHierarchy{K, L}([
        a.hierarchy[l] + b.hierarchy[l] for l in 1:(L+1)
    ])
end

function show(io::IO, ::MIME"text/plain", moment_hierarchy::MomentHierarchy{K, L}) where {K, L}
    for l in 0:L
        println(io, "Rank-$l tensors:")
        vecs = moment_hierarchy.hierarchy[l+1].moments
        for k in 0:K
            println(io, "\nOrder-(L=$l, K=$k):")
            show(io, "text/plain", vecs[k+1].tensor)
        end
    end
end

function moment_reconstruction(moments::MomentHierarchy{K, L}, fM) where {K, L}
    hats = []
    for l in 0:L
        for k in 0:K
            mlk = moments.hierarchy[l + 1].moments[k + 1]
            flk_hat = f_lk_hat(l, k, mlk, fM)
            push!(hats, flk_hat)
        end
    end
    return function(v)
        vhats = [hat(v) for hat in hats]
        result = []
        push!(result, sum(vhats))
        append!(result, vhats)
    end
end

end

module CollisionOperator

using ..Moments
using Memoization
using LinearAlgebra

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
    return (1/2)*sum([(factorial(big(k))*e(i + j))/factorial(big(j)) for j in 0:k]);
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

function collision_matrix_kernel(l, k)
    Float64.([A(l, q, m) + B(l, q, m) for m = 0:k, q = 0:k])
end

function laguerre_matrix(l, k)
    UpperTriangular(Float64.([Moments.c(l, q, m) for m=0:k, q=0:k]))
end

function C(l, p, k)
    sum(Moments.c(l, p, q) * Moments.c(l, k, m) * (A(l, q, m) + B(l, q, m)) for m=0:k, q=0:p)
end

@memoize Dict function collision_matrix(l, K)
    Float64.([C(l, p, k) for p=0:K, k=0:K])
end

@memoize Dict function collision_matrix_hierarchy(lmax, kmax)
    [collision_matrix(l, kmax) for l in 0:lmax]
end

function block_collision_matrix(L, K)
    matrix = zeros((L + 1) * (K + 1), (L + 1) * (K + 1))
    for l in 0:K
        range = (l * (K + 1) + 1):((l + 1) * (K + 1))
        display(range)
        matrix[range, range] .= collision_matrix(l, K)
    end
    matrix
end
end

function graph()
    x = x = range(-5, stop = 5, length = 100)
    f0 = Moments.f0
    f002 = Moments.maxwellian([2., 0., 0.], 1., 1.)
    bump_on_tail = v -> 0.9 * f0(v) + 0.1 * f002(v)
    flatten = (func) -> (vx -> func([vx, 0., 0.]))
    y1 = flatten(bump_on_tail).(x)

    scene = lines(x, y1, color = :blue)

    bump_on_tail_reconstruction_1 = Moments.moment_reconstruction(bump_on_tail, 1, 1)
    y2 = flatten(bump_on_tail_reconstruction_1).(x)
    bump_on_tail_reconstruction_2 = Moments.moment_reconstruction(bump_on_tail, 1, 8)
    y3 = flatten(bump_on_tail_reconstruction_2).(x)

    bump_on_tail_reconstruction_3 = Moments.moment_reconstruction(bump_on_tail, 3, 3)
    y4 = flatten(bump_on_tail_reconstruction_3).(x)


    lines!(x, y2, color = :red)
    lines!(x, y3, color = :green)
    lines!(x, y4, color = :purple)
end

function relaxation(l, k)
    collision_matrix = CollisionOperator.collision_matrix(l, k)
    return function update(moment_tensor)
        return moment_tensor * collision_matrix
    end
end

using DifferentialEquations

function test()
    f0 = Moments.f0
    f002 = Moments.maxwellian([0.5, 0., 0.])
    bump_on_tail = v -> 0.9 * f0(v) + 0.1 * f002(v)

    L = 1
    K = 3

    moments0 = Moments.moment_hierarchy(bump_on_tail, L, K)

end

function test2()
    L = 0
    K = 3
    moments0 = Moments.MomentHierarchy{K, L}([
        Moments.TensorMomentVector{K, L}([
            Moments.TensorMoment(fill(1.0, ())),
            Moments.TensorMoment(fill(-0.3, ())),
            Moments.TensorMoment(fill(1.1, ())),
            Moments.TensorMoment(fill(1.1, ())),
        ])
    ])

    (t, moments) = evolve(moments0; extra_k = 2)

    x = x = range(-5, stop = 5, length = 100)
    flatten = (func) -> (vx -> func([vx, 0., 0.]))

    scene = Scene()

    f = function(i)
        print(".")
        fv = Moments.moment_reconstruction(moments[i], Moments.fM)
        y = hcat(flatten(fv).(x)...)
        y
    end

    time = Node(1)
    fv = lift(i -> f(i), time)

    colors = [:blue, :red, :green, :purple, :yellow, :orange, :black]
    plots = []
    push!(plots, lines!(x, lift(f -> f[1,:], fv), color = colors[1])[end])
    for k in 0:(K + 2)
        p = lines!(
            x,
            lift(f -> f[k+2, :], fv),
            color = colors[k + 2],
        )
        push!(plots, p[end])
    end

    labels = ["Total"]
    append!(labels, ["k = $k" for k in 0:(K + 2)])
    center!(scene)
    lgd = legend(plots, labels, camera = campixel!, raw = true, strokecolor = colors)
    display(lgd)

    moment_plots = []
    for l in 0:L
        moment_plot = Scene()
        push!(moment_plots, moment_plot)
        for k in 0:(K + 2)
            foo::Array{Float64, 1} = repeat([NaN], length(t))
            leading_moments = lift(
                function(i)
                    foo[i] = log(norm(moments[i].hierarchy[1].moments[k + 1].tensor))
                    foo
                end,
            time)
            lines!(moment_plot, 1:length(t), leading_moments; limits = FRect(0, -10, length(t), 10), color = colors[k + 2])
        end
    end

    whole = vbox(scene, lgd, moment_plots...)

    record(whole, "test.mkv", 1:length(t); framerate = 12) do i
        push!(time, i)
    end
end

function evolve(moments0::Moments.MomentHierarchy{K, L}; extra_k = 0) where {K, L}
    solutions = []
    tspan = (0.0, 50.0)
    for l in 0:L
        lcat = (m1, m2) -> cat(m1, m2; dims=(l+1))
        moments = moments0.hierarchy[l + 1].moments
        extra_zero_moments = repeat([zeros(size(moments[1].tensor))], extra_k)
        m0 = reduce(lcat, append!([m.tensor for m in moments], extra_zero_moments))
        display(m0)
        col = CollisionOperator.collision_matrix(l, K + extra_k)
        display(col)

        colons = repeat([:], l)
        display(view(m0, colons..., 2))

        result = reduce(lcat, [
            sum(view(m0, colons..., j) .* col[i, j] for j in 1:size(m0)[1])
            for i in 1:size(col)[1]
        ])
        display(result)

        prob = ODEProblem((m, p, t) ->
                          reduce(lcat, [
                              sum(view(m, colons..., j) .* col[i, j] for j in 1:size(m)[1])
                              for i in 1:size(col)[1]
                          ]),
                          m0, tspan)
        sol = DifferentialEquations.solve(prob, alg=DP5())
        push!(solutions, sol)
    end

    dt = 0.5

    ts = Array(tspan[1]:dt:tspan[2])
    hierarchies = []
    for t in ts
        hierarchy = []
        for l in 0:L
            colons = repeat([:], l)
            sol = solutions[l + 1]
            m = sol(t)
            moment_vec::Array{Moments.TensorMoment{L}, 1} = []
            for k in 0:(K + extra_k)
                push!(moment_vec, Moments.TensorMoment{L}(Array(view(m, colons..., k+1))))
            end
            push!(hierarchy, Moments.TensorMomentVector{K + extra_k, L}(moment_vec))
        end
        push!(hierarchies, Moments.MomentHierarchy{K + extra_k, L}(hierarchy))
    end
    display(hierarchies[3])

    ts, hierarchies
end

function antidiag(n)
    mm = diagm(ones(n))
    hcat([mm[n + 1 - i,:] for i in 1:n]...)
end

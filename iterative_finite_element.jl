using Plots
using GaussQuadrature
using SparseArrays
using BenchmarkTools
using .Threads
using LinearAlgebra

function gauss_jacobi(A::AbstractMatrix{T}, b::AbstractVector{T}, tol::T, max_iter::Int=10000000) where T<:Real
  n = length(b)
  x = zeros(n)  # Initial guess
  x_new = zeros(n)

  for k in 1:max_iter
    # i = 1
    sum_Ax_1 = 0.0
    for j in 2:n
      sum_Ax_1 += A[1, j] * x[j]
    end
    x_new[1] = (b[1] - sum_Ax_1) / A[1, 1]

    for i in 2:n-1
      sum_Ax = 0.0
      for j in 1:n
        if j != i
          sum_Ax += A[i, j] * x[j]
        end
      end
      x_new[i] = (b[i] - sum_Ax) / A[i, i]
    end

    # i = n
    sum_Ax_n = 0.0
    for j in 1:n-1
      sum_Ax_n += A[n, j] * x[j]
    end
    x_new[n] = (b[n] - sum_Ax_n) / A[n, n]

    # tol reached
    if norm(x_new - x, Inf) < tol
      return x_new
    end
    x .= x_new
  end

  # max_iter reached
  return x_new
end

function gauss_jacobi_parallel(A::AbstractMatrix{T}, b::AbstractVector{T}, tol::T, max_iter::Int=10000000) where T<:Real
  n = length(b)
  x = zeros(n)
  x_new = zeros(n)

  for k in 1:max_iter
    @threads for i in 1:n
      sum_Ax = 0.0
      for j in 1:n
        if j != i
          sum_Ax += A[i, j] * x[j]
        end
      end
      x_new[i] = (b[i] - sum_Ax) / A[i, i]
    end

    if norm(x_new - x, Inf) < tol
      return x_new
    end

    x .= x_new
  end

  return x_new
end

function gauss_jacobi_tridiagonal(A::AbstractMatrix{T}, b::AbstractVector{T}, tol::T, max_iter::Int=10000000) where T<:Real
  n = length(b)
  x = zeros(n)  # Initial guess
  x_new = zeros(n)

  for iter in 1:max_iter
    # i = 1
    x_new[1] = (b[1] - A[1, 2] * x[2]) / A[1, 1]

    # i in 2:n-1
    for i in 2:n-1
      x_new[i] = (b[i] - A[i, i-1] * x[i-1] - A[i, i+1] * x[i+1]) / A[i, i]
    end

    # i = n
    x_new[n] = (b[n] - A[n, n-1] * x[n-1]) / A[n, n]

    # tol reached
    if norm(x_new - x, Inf) < tol
      return x_new
    end

    x .= x_new
  end

  # max_iter reached
  return x_new
end

function gauss_jacobi_tridiagonal_parallel(A::AbstractMatrix{T}, b::AbstractVector{T}, tol::T, max_iter::Int=10000000) where T<:Real
  n = length(b)
  x = zeros(n)
  x_new = zeros(n)

  for iter in 1:max_iter
    # i = 1
    x_new[1] = (b[1] - A[1, 2] * x[2]) / A[1, 1]

    @threads for i in 2:n-1
      x_new[i] = (b[i] - A[i, i-1] * x[i-1] - A[i, i+1] * x[i+1]) / A[i, i]
    end

    x_new[n] = (b[n] - A[n, n-1] * x[n-1]) / A[n, n]

    if norm(x_new - x, Inf) < tol
      return x_new
    end

    x .= x_new
  end

  return x_new
end

function gauss_seidel(A::AbstractMatrix{T}, b::AbstractVector{T}, tol::T, max_iter::Int=10000000) where T<:Real
  n = size(A, 1)
  x = zeros(n)

  for k in 1:max_iter
    max_diff = 0.0

    for i in 1:n
      sum1 = i > 1 ? sum(A[i, j] * x[j] for j in 1:i-1) : 0.0
      sum2 = i < n ? sum(A[i, j] * x[j] for j in i+1:n) : 0.0
      x_new = (b[i] - sum1 - sum2) / A[i, i]
      max_diff = max(max_diff, abs(x_new - x[i]))
      x[i] = x_new
    end

    if max_diff < tol
      return x
    end
  end

  return x
end

function gauss_seidel_tridiagonal(A::AbstractMatrix{T}, b::AbstractVector{T}, tol::T, max_iter::Int=10000000) where T<:Real
  n = length(b)
  x = zeros(n)  # Initial guess

  for iter in 1:max_iter
    max_diff = 0.0

    # i = 1
    old_x1 = x[1]
    x[1] = (b[1] - A[1, 2] * x[2]) / A[1, 1]
    max_diff = max(max_diff, abs(x[1] - old_x1))

    # i from 2 to n-1
    for i in 2:n-1
      old_xi = x[i]
      x[i] = (b[i] - A[i, i-1] * x[i-1] - A[i, i+1] * x[i+1]) / A[i, i]
      max_diff = max(max_diff, abs(x[i] - old_xi))
    end

    # i = n
    old_xn = x[n]
    x[n] = (b[n] - A[n, n-1] * x[n-1]) / A[n, n]
    max_diff = max(max_diff, abs(x[n] - old_xn))

    if max_diff < tol
      return x
    end
  end

  return x
end





# Approximates the Integral of a given function in the interval [-1:1]
function gaussian_quadrature(f, ngp)

  # Initializing P and W according to the number of Gauss points
  P, W = legendre(ngp)
  sum = 0

  for j in 1:ngp
    sum = sum + W[j] * f(P[j])
  end

  return sum
end

# Initializes the LG matrix
function init_LG_matrix(ne)
  LG = zeros(Int, 2,ne)

    for j in 1:ne
      LG[1,j] = j
      LG[2,j] = j + 1
    end

  return LG
end

# Initializes the EQ vector and the m variable
function init_EQ_vector_and_m(ne)
  # Initializing m and EQ
  m = ne - 1
  EQ = zeros(Int, ne+1)

  # Computing the first element of EQ
  EQ[1] = m + 1

  # Computing the mid elements of EQ
  for i in 1:m+1
    EQ[i+1] = i
  end

  # Computing the last element of EQ
  EQ[ne+1] = m + 1

  return EQ, m
end

# Initializes the K matrix
function init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)

  # Initializes the Ke matrix
  function init_Ke_matrix(ne, alpha, beta, gamma)
    h = 1 / ne
    Ke = zeros(2,2)

    for a in 1:2
      for b in 1:2
        Ke[a,b] = (alpha * 2 / h) * gaussian_quadrature((qsi) -> d_phi(a, qsi) * d_phi(b, qsi), 2) + (beta * h / 2) * gaussian_quadrature((qsi) -> phi(a, qsi) * phi(b, qsi), 2) + gamma * gaussian_quadrature((qsi) -> d_phi(b, qsi) * phi(a, qsi), 2)
      end
    end

    return Ke
  end

  # Initializing K and Ke matrices
  K = spzeros(m+1,m+1)
  Ke = init_Ke_matrix(ne, alpha, beta, gamma)

  for e in 1:ne
    for a in 1:2
      i = Int(EQ[LG[a, e]])
      for b in 1:2
        j = Int(EQ[LG[b, e]])
        K[i,j] += Ke[a,b]
      end
    end
  end

  # removes the last line and column
  return K[1:m, 1:m]
end

# Initializes the F vector
function init_F_vector(f, ne, EQ, LG, m)

  # Initializes the Fe vector
  function init_Fe_vector(f, ne, e)
    Fe = zeros(2)
    h = 1 / ne

    for a in 1:2
      Fe[a] = (h / 2) * gaussian_quadrature((qsi) -> f(qsi_to_x(qsi, e, h)) *  phi(a, qsi), 5)
    end

    return Fe
  end

  # Initializing the F vector and the variable h
  h = 1 / ne
  F = zeros(m+1)

  for e in 1:ne
    Fe = init_Fe_vector(f, ne, e)
    for a in 1:2
      i = EQ[LG[a,e]]
      F[i] += Fe[a]
    end
  end

  # removes the last line
  return F[1:m]
end

# Generalized phi function
function phi(number, qsi)
  return [((1 - qsi) / 2), ((1 + qsi) / 2)][number]
end

# Generalized derivative of the phi function
function d_phi(number, qsi)
  return [(-1 / 2), (1 / 2)][number]
end

# Converts the interval from [x_i-1 , xi+1] to [-1, 1]
function qsi_to_x(qsi, i, h)
  return (h / 2) * (qsi + 1) + 0 + (i - 1)*h
end

function solve_system(ne, alpha, beta, gamma, f, u, solver, tol)
  # Initializing matrices, vectors and variables
  EQ, m = init_EQ_vector_and_m(ne)
  LG    = init_LG_matrix(ne)
  K     = init_K_matrix(ne, EQ, LG, alpha, beta, gamma, m)
  F     = init_F_vector(f, ne, EQ, LG, m)
  return solver(K, F, tol)
end

# Plots the exact and inexact graphs, as well as the absolute and relative errors
function plot_comparison(ne, alpha, beta, gamma, f, u, solver, tol)
  # Initializing variables
  h = 1 / ne
  xs = [h * i for i in 1:ne-1]
  Cs = solve_system(ne, alpha, beta, gamma, f, u, solver, tol)

  # Including the boundary conditions in both xs and Cs
  ext_xs = [0; xs; 1]
  ext_Cs = [0; Cs; 0]

  # Plotting the exact function and our approximation
  plt = plot(u, 0, 1, label = "u(x)", size=(800, 800))
  plot!(plt, ext_xs, ext_Cs, seriestype = :scatter, label = "Approximation", xlabel = "x", ylabel = "Approximation for u(x)", size=(800, 800))

  # Saving the graph
  savefig("approximation-graph.png")
end

# Plots the graph of errors according to the varying of n
function error_analysis(lb, ub, method, name, tol)

  # Computes the error according to ne
  function gauss_error(u, cs, ne, EQ, LG)
    sum = 0
    h = 1 / ne

    # including 0 so that the EQ-LG will not consider the first and the last phi function
    extended_cs = [cs; 0]

    # Computing the error
    for e in 1:ne
      sum = sum + gaussian_quadrature((qsi) -> (u(qsi_to_x(qsi, e, h)) - (extended_cs[EQ[LG[1,e]]] * phi(1, qsi)) - (extended_cs[EQ[LG[2,e]]] * phi(2, qsi)))^2, 5)
    end

    return sqrt(sum * (h / 2))
  end

  # Initializing the vectors
  errors = zeros(ub - lb + 1)
  nes = [(1 << i) - 1 for i in lb:ub]
  hs = [1 / nes[i - lb + 1] for i in lb:ub]

  # Computing the errors varying according to the variation of h
  for i in lb:ub
    display("potência de 2 = $i")
    ne = nes[i-lb+1]
    EQ, m = init_EQ_vector_and_m(ne)
    LG = init_LG_matrix(ne)
    Cs = solve_system(ne, alpha, beta, gamma, f, u, method, tol[i])
    e = gauss_error(u, Cs, ne, EQ, LG)
    errors[i-lb+1] = e
  end

  # Plotting the errors in the graphic in a log scale
  plot(hs, errors, seriestype = :scatter, label = "Error convergence ",
       xlabel = "h", ylabel = "error", size=(800, 800), xscale=:log10, yscale=:log10,
       markercolor = :blue)
  plot!(hs, errors, seriestype = :line, label = "", linewidth = 2, linecolor = :blue)
  plot!(hs, hs.^2, seriestype = :line, label = "h^2", linewidth = 2, linecolor = :red)

  # Saves the graph in a png file
  savefig("errors-convergence-$name.png")
end


function compare_methods_in_solving_system_varying_ne(lb, ub, alpha, beta, gamma, f, u, methods, names, markershapes, tols)
  nes = [2^i for i in lb:ub]
  n = length(names)
  k = length(nes)
  times = [Vector{Float64}() for _ in 1:n]

  for j in 1:k
    println("índice do ne = $j")
    for i in 1:n
      nome = names[i]
      println("método = $nome")
      t = @benchmark $solve_system($nes[$j], $alpha, $beta, $gamma, $f, $u, $methods[$i], $tols[Int(round(log(2, $nes[$j])))])
      t_min = minimum(t.times) / 1e9
      push!(times[i], t_min)
    end
  end

  plot(title="Tempo de cada método", ylabel="tempo (s)", xlabel="n_e",
       legend=:topleft, size=(800, 800), yscale=:log10, xscale=:log2)

  for i in 1:n
    plot!(nes, times[i], label=names[i], linewidth=2, markershape=markershapes[i], markersize=8)
  end

  savefig("compare-methods-in-solving-system-varying-ne-24-threads-$lb-to-$ub.png")
end


# Compares numerical methods by the time they took to compute the error convergency
function compare_methods_in_error_convergence(lb, ub, methods, names, tols)
  n = length(methods)
  for i in 1:n
    nome = names[i]
    println("método = $nome")
    error_analysis(lb, ub, methods[i], names[i], tols)
  end
end


# Methods
methods = [
  (A, b, t) -> A \ b;
  (A, b, t) -> gauss_jacobi(A, b, t);
  (A, b, t) -> gauss_jacobi_parallel(A, b, t);
  (A, b, t) -> gauss_jacobi_tridiagonal(A, b, t);
  (A, b, t) -> gauss_jacobi_tridiagonal_parallel(A, b, t);
  (A, b, t) -> gauss_seidel(A, b, t);
  (A, b, t) -> gauss_seidel_tridiagonal(A, b, t)
]

# Methods names
names = [
  "Julia backslash";
  "Gauss Jacobi";
  "Gauss Jacobi Paralelo";
  "Gauss Jacobi Tri-diagonal";
  "Gauss Jacobi Tri-diagonal Paralelo";
  "Gauss Seidel";
  "Gauss Seidel Tri-diagonal"
]

# Shape for each method in plotting
markershapes = [
  :circle;
  :rect;
  :hexagon;
  :utriangle;
  :pentagon;
  :star5;
  :vline
]

# Constants
alpha = 1
beta  = 1
gamma = 1

# Functions
f = (x) -> alpha*pi^2 * sin(pi*x) + beta*sin(pi*x) + gamma*pi*cos(pi*x)
u = (x) -> sin(pi * x)

# Bound limits for analyzing the error convergence
lb = 2
ub = 4

                #2    #3    #4    #5    #6    #7    #8     #9     #10
tolerance = [0; 1e-4; 1e-5; 1e-6; 1e-7; 1e-8; 1e-9; 1e-10; 1e-12; 1e-13;]



compare_methods_in_error_convergence(lb, ub, methods, names, tolerance)

compare_methods_in_solving_system_varying_ne(lb, ub, alpha, beta, gamma, f, u, methods, names, markershapes, tolerance)

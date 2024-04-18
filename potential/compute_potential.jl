raw"""
Aziz's HFDHE2 potential energy function

This module provides the potential energy function and its gradient for the HFDHE2 potential energy function.

The potential energy function is given by:

$$
V(r) = \epsilon \cdot V^*(r/r_m)
$$

where $V^*(r)$ is given by:

$$
V^*(r) = A \cdot \exp(-\alpha r) - \left(\frac{C_6}{r^6} + \frac{C_8}{r^8} + \frac{C_{10}}{r^{10}}\right) \cdot F(r)
$$

and $F(r)$ is given by:

$$
F(r) = \begin{cases}
1 & \text{if } r \geq D \\
\exp\left(-\left(\frac{D}{r} - 1\right)^2\right) & \text{otherwise}
\end{cases}
$$
"""

using LinearAlgebra
using ForwardDiff
using DiffResults

const A = 0.5448506e6
const α = 13.353384
const C6 = 1.3732412
const C8 = 0.4253785
const C10 = 0.178100
const D = 1.241314
const rm = 2.9673 / 0.529177 # In atomic units
const kb = 3.166811563e-6
const ϵ_K = 10.8
const ϵ = ϵ_K * kb # In atomic units

function F(x :: T) :: T where T
    return x >= D ? one(T) : exp( - (D / x - 1)^2 )
end

function V_star(x :: T) :: T where T
    ret = A * exp(-α * x)
    ret -= (C6 / x^6 + C8 / x^8 + C10 / x^10) * F(x)
    return ret
end


function V(r :: T) :: T where T
    return ϵ * V_star(r / rm)
end

function potential_energy(coords :: Matrix{T}) :: T where {T}
    potential = zero(T)

    _, n_atoms = size(coords)

    for i in 1:n_atoms 
        for j in i+1:n_atoms
            r = norm(coords[:, i] - coords[:, j])
            potential += V(r)
        end
    end
    return potential
end

# Get the gradient of the potential energy using ForwardDiff
function energy_forces(coords :: Matrix{T})  where {T}
    forces = zeros(T, size(coords))
    energy = energy_forces!(forces, coords)
    return energy, forces
end
function energy_forces!(forces :: Matrix{T}, coords :: Matrix{T}) :: T where {T}
    # Define the diff result type
    energy = zero(T)
    grad_result = DiffResults.DiffResult(energy, forces)

    ForwardDiff.gradient!(grad_result, potential_energy, coords)

    #@views forces[:, :] = -DiffResults.gradient(grad_result)
    @views forces[:, :] *= -1

    #forces *= -1
    return DiffResults.value(grad_result)
end

function test_potential_forces()
    coords = zeros(Float64, (3, 2))
    r_vals = 4.0:0.01:6.0
    energies = zeros(Float64, length(r_vals))
    forces = zeros(Float64, length(r_vals))
    all_forces = zeros(Float64, 3, 2)

    for i in 1:length(r_vals)
        r = r_vals[i]
        coords[1, 2] = r
        energy = energy_forces!(all_forces, coords)
        energies[i] = energy
        forces[i] = all_forces[1, 2]
    end

    return r_vals, energies, forces
end


@doc raw"""
    compute_ensemble!(energies, forces, coords)

Compute the potential energy and forces for a set of configurations.
"""
function compute_ensemble!(energies :: AbstractArray{T}, forces :: Matrix{T}, coords :: Matrix{T}) where {T}
    _, n_atoms, n_configs = size(coords)
    
    # Assert that the dimensions of the arrays are correct
    @assert size(energies) == (n_configs,)
    @assert size(forces) == (3, n_atoms, n_configs)

    for i in 1:n_configs 
        @views energies[i] = energy_forces!(forces[:, :, i], coords[:, :, i])
    end
end

function compute_ensemble_py(coords_py)
    coords = permutedims(coords_py, [3, 2, 1])
    n_atoms = size(coords, 2)
    n_configs = size(coords, 3)

    # Convert Angstrom to atomic units
    @views coords[:, :, :] /= 0.529177

    energies = zeros(eltype(coords), n_configs)
    forces = zeros(eltype(coords), 3, n_atoms, n_configs)

    compute_ensemble!(energies, forces, coords)

    # Convert energies from Ha to Ry
    @views energies[:] *= 2

    # Convert the forces from Ha/Bohr to Ry/Angstrom
    @views forces[:, :, :] *= 2 * 0.529177

    return energies, permutedims(forces, [3, 2, 1])
end

# Reference values for Statement #87 (and the LogSumExp operator of #88) from
# entrain's own Julia implementation.
#
# Regenerate the fixture this file produces:
#   julia --startup-file=no tests/fixtures/sterility_reference.jl \
#         > tests/fixtures/sterility_reference.json
#
# The default path assumes the AXIS workspace layout (ENRG-AI next to entrain).
# Point ENTRAIN_JL at consistency.jl to read another checkout:
#   ENTRAIN_JL=/path/to/consistency.jl julia --startup-file=no ...
#
# consistency.jl guards its report behind `PROGRAM_FILE == @__FILE__`, so
# including it here only defines the functions — nothing is simulated by
# accident. One relief trace is lived here (seed 1, ε = 1e-3), and the port in
# agent/physics/sterility.py must read the same A² and CV off the same numbers.
const SRC = get(
    ENV,
    "ENTRAIN_JL",
    joinpath(@__DIR__, "..", "..", "..", "entrain", "julia", "kuramoto", "consistency.jl"),
)
include(normpath(SRC))

floats(x) = "[" * join([string(v) for v in x], ",") * "]"

trace = relief_trace(1e-3; seed = 1)
index = sterility_index(trace)

# The contraction of Statement #88: three rows h_k(λ) = a_k + b_k·λ with
# max|b_k| = 0.5, and the fixed point of the unsmoothed operator.
lift = (0.0, 1.0, 2.0)
slope = (0.5, 0.2, -0.1)
rows_at = λ -> [lift[k] + slope[k] * λ for k in 1:3]
root = banach_iteration(λ -> maximum(rows_at(λ)) - 0.8, 1.0)

println("{")
println("\"julia\": \"$(VERSION)\",")
println("\"source\": \"entrain/julia/kuramoto/consistency.jl\",")
println("\"statement\": 87,")
println("\"memory\": $(trace.memory),")
println("\"times\": $(floats(trace.times)),")
println("\"relief\": $(floats(trace.relief)),")
println("\"readings\": {")
println("  \"a2\": $(index.a2),")
println("  \"cv\": $(index.cv),")
println("  \"r0\": $(index.r0),")
println("  \"verdict\": \"$(index.verdict)\"")
println("},")
println("\"smooth_max\": {\"values\": $(floats([1.0, 0.9, 0.8])), ")
println("  \"beta16\": $(smooth_max([1.0, 0.9, 0.8], 16.0)), ")
println("  \"beta1\": $(smooth_max([1.0, 0.9, 0.8], 1.0))},")
println("\"banach\": {\"lift\": $(floats(collect(lift))), ")
println("  \"slope\": $(floats(collect(slope))), \"lam\": $(root.lam)}")
println("}")

# Reference values from entrain's own Julia implementation.
#
# Regenerate the fixture this file produces:
#   julia --startup-file=no tests/fixtures/entrain_reference.jl \
#         > tests/fixtures/entrain_reference.json
#
# The default path assumes the AXIS workspace layout (ENRG-AI next to entrain).
# Point ENTRAIN_JL at basic.jl to read another checkout:
#   ENTRAIN_JL=/path/to/basic.jl julia --startup-file=no ...
#
# basic.jl guards its demo behind `PROGRAM_FILE == @__FILE__`, so including it
# here only defines the functions — nothing is simulated by accident.
const SRC = get(
    ENV,
    "ENTRAIN_JL",
    joinpath(@__DIR__, "..", "..", "..", "entrain", "julia", "kuramoto", "basic.jl"),
)
include(normpath(SRC))

floats(x) = "[" * join([string(v) for v in x], ",") * "]"
mat(M) = "[" * join([floats(M[i, :]) for i in 1:size(M, 1)], ",") * "]"

omega0 = [1.0 2.0; 1.1 2.2; 0.9 1.8]
theta = [0.0 0.3; 1.0 1.6; 2.0 2.7]
coupling = [0.0 0.5 0.4; 0.5 0.0 0.3; 0.4 0.3 0.0]
m = Medium(omega0 = omega0, theta = theta, coupling = coupling,
           band = 0.1, link = 0.6, degree = 3)

println("{")
println("\"julia\": \"$(VERSION)\",")
println("\"source\": \"entrain/julia/kuramoto/basic.jl\",")
println("\"inputs\": {")
println("  \"omega0\": $(mat(omega0)),")
println("  \"theta\": $(mat(theta)),")
println("  \"coupling\": $(mat(coupling)),")
println("  \"band\": 0.1, \"link\": 0.6, \"degree\": 3,")
println("  \"memory\": 0.5, \"damping\": 0.8, \"tuning\": 0.0, \"dt\": 0.01")
println("},")
println("\"readings\": {")
println("  \"coherence\": $(coherence(m)),")
println("  \"partial_coherence\": $(floats([partial_coherence(m, p) for p in 1:2])),")
println("  \"resonance\": $(resonance(m)),")
println("  \"spread\": $(floats([spread(m, i) for i in 1:3])),")
println("  \"chord\": $(floats([chord(m, i, 0.7) for i in 1:3])),")
println("  \"wave\": $(wave(m, 0.7)),")
println("  \"unison\": $(floats([unison(m, i) for i in 1:3])),")
println("  \"coincidence\": $(floats([coincidence(m, i) for i in 1:3]))")
println("},")
print("\"memory_goal\": [")
goal(n, acc, weight) = begin
    print("$(n == 1 ? "" : ","){\"acc\": $acc, \"weight\": $weight, \"i\": 1, \"j\": 2, \"value\": $(memory_goal(m, 1, 2, acc, weight))}")
end
goal(1, 0.0, 1.0); goal(2, 1.0, 1.0); goal(3, 0.5, 2.0); goal(4, 2.0, 0.0)
println("],")
print("\"overlap\": [")
for (n, (d, b)) in enumerate([(0.0, 0.1), (0.1, 0.1), (0.2, 0.1), (0.05, 0.1), (0.0, Inf)])
    n > 1 && print(",")
    bs = isinf(b) ? "\"inf\"" : string(b)
    print("{\"delta\": $d, \"band\": $bs, \"value\": $(overlap(d, b))}")
end
println("],")
# Three steps of the full law: frequencies, then phases, then the memory write.
m2 = Medium(omega0 = omega0, theta = theta, coupling = coupling,
            band = 0.1, link = 0.6, degree = 3, memory = 0.5,
            damping = 0.8, tuning = 0.0)
print("\"steps\": [")
for n in 1:3
    step!(m2, 0.01)
    n > 1 && print(",")
    print("{\"omega\": $(mat(m2.omega)), \"theta\": $(mat(m2.theta)), \"coupling\": $(mat(m2.coupling))}")
end
println("]")
println("}")

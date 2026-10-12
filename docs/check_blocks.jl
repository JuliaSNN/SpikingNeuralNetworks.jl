# Run the ```julia code blocks of documentation pages that Documenter does not execute.
#
#   julia --project=docs -t 1 docs/check_blocks.jl                 # catalogue, tutorials, models_ext
#   julia --project=docs -t 1 docs/check_blocks.jl page.md ...     # given pages
#   julia --project=docs -t 1 docs/check_blocks.jl --figures       # also rewrite the tutorial figures
#
# A block preceded by `<!-- norun -->` is skipped. Blocks of a page containing `<!-- run: sequential -->`
# share one module; otherwise every block runs in a fresh module. Blocks run in a temporary directory,
# except with `--figures`, where the tutorial pages run in `src/assets/tutorials` so that their `save`
# calls regenerate the figures. Exits with status 1 if a block throws.
using SpikingNeuralNetworks, CairoMakie

const SRC = joinpath(@__DIR__, "src")

function blocks(path)
    lines = readlines(path)
    out = Tuple{Int,String}[]
    i, prev = 1, ""
    while i <= length(lines)
        l = lines[i]
        if strip(l) == "```julia"
            ind = length(l) - length(lstrip(l))
            start, buf = i, String[]
            i += 1
            while i <= length(lines) && strip(lines[i]) != "```"
                push!(buf, length(lines[i]) >= ind ? lines[i][ind+1:end] : lstrip(lines[i]))
                i += 1
            end
            occursin("norun", prev) || push!(out, (start, join(buf, "\n")))
        end
        isempty(strip(l)) || (prev = strip(l))
        i += 1
    end
    return out, any(l -> occursin("run: sequential", l), lines)
end

function run_page(path, dir)
    bs, seq = blocks(path)
    nfail = 0
    m = Module()
    cd(dir) do
        for (ln, code) in bs
            seq || (m = Module())
            Core.eval(m, :(using SpikingNeuralNetworks, CairoMakie))
            t = time()
            try
                redirect_stdout(devnull) do
                    Base.include_string(m, code, "$path:$ln")
                end
                println("ok   $(relpath(path, SRC)):$ln  $(round(time() - t, digits = 1)) s")
            catch e
                nfail += 1
                println("FAIL $(relpath(path, SRC)):$ln  ", first(sprint(showerror, e), 400))
            end
        end
    end
    return nfail
end

figures = "--figures" in ARGS
pages = filter(a -> a != "--figures", ARGS)
if isempty(pages)
    pages = vcat(readdir(joinpath(SRC, "catalogue"), join = true), readdir(joinpath(SRC, "tutorials"), join = true),
                 [joinpath(SRC, "models_ext.md")])
    pages = filter(endswith(".md"), pages)
end
tmp = mktempdir()
nfail = sum(pages; init = 0) do p
    dir = figures && occursin("tutorials", p) ? joinpath(SRC, "assets", "tutorials") : tmp
    run_page(abspath(p), dir)
end
println("failures: $nfail")
exit(nfail == 0 ? 0 : 1)

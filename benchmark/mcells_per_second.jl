using Printf, Statistics

function export_markdown_mcells(filename, results)
    nthreads = results.benchmarkgroup["2DAcoIsoDenQ_DEO2_FDTD, no IO, forward"].tags[1]["nthreads"]

    rows = [
        "2DAcoIsoDenQ_DEO2_FDTD, no IO, forward", "2DAcoIsoDenQ_DEO2_FDTD, forward", "2DAcoIsoDenQ_DEO2_FDTD, linear", "2DAcoIsoDenQ_DEO2_FDTD, adjoint",
        "2DAcoVTIDenQ_DEO2_FDTD, no IO, forward", "2DAcoVTIDenQ_DEO2_FDTD, forward", "2DAcoVTIDenQ_DEO2_FDTD, linear", "2DAcoVTIDenQ_DEO2_FDTD, adjoint",
        "2DAcoTTIDenQ_DEO2_FDTD, no IO, forward", "2DAcoTTIDenQ_DEO2_FDTD, forward", "2DAcoTTIDenQ_DEO2_FDTD, linear", "2DAcoTTIDenQ_DEO2_FDTD, adjoint",
        "3DAcoIsoDenQ_DEO2_FDTD, no IO, forward", "3DAcoIsoDenQ_DEO2_FDTD, forward", "3DAcoIsoDenQ_DEO2_FDTD, linear", "3DAcoIsoDenQ_DEO2_FDTD, adjoint",
        "3DAcoVTIDenQ_DEO2_FDTD, no IO, forward", "3DAcoVTIDenQ_DEO2_FDTD, forward", "3DAcoVTIDenQ_DEO2_FDTD, linear", "3DAcoVTIDenQ_DEO2_FDTD, adjoint",
        "3DAcoTTIDenQ_DEO2_FDTD, no IO, forward", "3DAcoTTIDenQ_DEO2_FDTD, forward", "3DAcoTTIDenQ_DEO2_FDTD, linear", "3DAcoTTIDenQ_DEO2_FDTD, adjoint"]
    columns = ["$i threads" for i in nthreads]

    μ = zeros(length(rows), length(nthreads))
    σ = zeros(length(rows), length(nthreads))

    for (iprop,prop) in enumerate(rows)
        ncells = results.benchmarkgroup[prop].tags[1]["ncells"]
        for (ithread,nthreads) in enumerate(columns)
            benchmark = results.benchmarkgroup[prop][nthreads]
            x = (ncells / 1_000_000) ./ (benchmark.times .* 1e-9) # Mega-Cells per second
            μ[iprop,ithread] = mean(x)
            σ[iprop,ithread] = std(x)
        end
    end

    io = open(filename, "w")

    write(io, "# JetPackWaveFD Operator Throughput\n\n")

    write(io, "|    ")
    for column in columns
        write(io, " | $column")
    end
    write(io, "|\n")
    write(io, "|------")
    for column in columns
        write(io, "| ------ ")
    end
    write(io, "|\n")
    for (irow,row) in enumerate(rows)
        write(io, "| $row")
        for icol = 1:length(columns)
            _μ = @sprintf("%.2f", μ[irow,icol])
            _σ = @sprintf("%.2f", 100* (σ[irow,icol] / μ[irow,icol]))
            write(io, " | $_μ MC/s ($_σ %)")
        end
        write(io, "|\n")
    end

    write(io, "\n## Julia versioninfo\n")
    write(io, "```\n")
    write(io, results.vinfo)
    write(io,"```\n")
    close(io)
end

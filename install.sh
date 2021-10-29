
# Julia dependencies
# install Julia packages in /opt/julia instead of $HOME
JULIA_DEPOT_PATH=/opt/julia
JULIA_PKGDIR=/opt/julia
JULIA_VERSION=1.3.1

mkdir /opt/julia-${JULIA_VERSION} && \
    cd /tmp && \
    wget -q https://julialang-s3.julialang.org/bin/linux/x64/`echo ${JULIA_VERSION} | cut -d. -f 1,2`/julia-${JULIA_VERSION}-linux-x86_64.tar.gz && \
    echo "faa707c8343780a6fe5eaf13490355e8190acf8e2c189b9e7ecbddb0fa2643ad *julia-${JULIA_VERSION}-linux-x86_64.tar.gz" | sha256sum -c - && \
    tar xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt/julia-${JULIA_VERSION} --strip-components=1 && \
    rm /tmp/julia-${JULIA_VERSION}-linux-x86_64.tar.gz
    ln -fs /opt/julia-*/bin/julia /usr/local/bin/julia




mkdir /etc/julia &&     echo "push!(Libdl.DL_LOAD_PATH, \"$CONDA_DIR/lib\")" >> /etc/julia/juliarc.jl 
# conda clean --all -f -y 
julia -e 'import Pkg; Pkg.update()' &&    julia -e 'import Pkg; Pkg.add("HDF5")' &&     julia -e "using Pkg; pkg\"add IJulia\"; pkg\"precompile\"" &&     julia -e "using Pkg; pkg\"add PyCall \"; pkg\"precompile\"" 

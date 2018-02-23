set -ex

export CI=true

export TOP_DIR=$(dirname $(dirname $(readlink -e "${BASH_SOURCE[0]}")))

export OS="$(uname)"


compilers=(
    cc
    c++
    gcc
    g++
    x86_64-linux-gnu-gcc
)

# setup ccache
if [[ "$OS" == "Darwin" ]]; then
    export PATH="/usr/local/opt/ccache/libexec:$PATH"
else
    if [[ -n "${SCCACHE_BUCKET}" ]]; then
        if hash sccache 2>/dev/null; then
            echo "SCCACHE_BUCKET is set but sccache executable is not found"
            exit 1
        fi
        SCCACHE_BIN_DIR="$TOP_DIR/sccache"
        mkdir -p "$SCCACHE_BIN_DIR"
        for compiler in "${compilers[@]}"; do
            (
                echo "#!/bin/sh"
                echo "exec $SCCACHE $(which $compiler) \"\$@\""
            ) > "$SCCACHE_BIN_DIR/$compiler"
            chmod +x "$SCCACHE_BIN_DIR/$compiler"
        done
        export PATH="$SCCACHE_BIN_DIR:$PATH"
    else
        if [[ -d "/usr/lib/ccache" ]]; then
            export PATH="/usr/lib/ccache:$PATH"
        elif hash ccache 2>/dev/null; then
            CCACHE_BIN_DIR="$TOP_DIR/ccache"
            mkdir -p "$CCACHE_BIN_DIR"
            for compiler in "${compilers[@]}"; do
                ln -sf "$(which ccache)" "$CCACHE_BIN_DIR/$compiler"
            done
            export PATH="$CCACHE_BIN_DIR:$PATH"
        fi
    fi
fi

# setup virtualenv
virtualenv "$TOP_DIR/venv"
source "$TOP_DIR/venv/bin/activate"
pip install -U pip setuptools

FROM python:3.14-slim-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# -------------------------
# システム依存パッケージ
# -------------------------
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    gfortran \
    git \
    git-lfs \
    openssh-client \
    ca-certificates \
    curl \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# WSL ユーザーに合わせたユーザー作成 (UID/GID = 1000)
# -------------------------
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd \
        --uid ${USER_UID} \
        --gid ${USER_GID} \
        --shell /bin/bash \
        --create-home \
        ${USERNAME} \
    # パスワードなし sudo 権限 (パッケージインストール用)
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# -------------------------
# GitHub の SSH ホスト鍵を事前登録 (接続時の確認プロンプトを回避)
# -------------------------
RUN mkdir -p /home/${USERNAME}/.ssh \
    && ssh-keyscan -H github.com >> /home/${USERNAME}/.ssh/known_hosts \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.ssh \
    && chmod 700 /home/${USERNAME}/.ssh \
    && chmod 600 /home/${USERNAME}/.ssh/known_hosts

# -------------------------
# Node.js (Node 24系) インストール
# Claude Code CLI自体はdevcontainer.jsonのfeature（ghcr.io/anthropics/devcontainer-features/claude-code）
# が入れるため、ここでは重複させずNode.jsのみ導入する。
# -------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# R インストール（ベンチマーク用リファレンス実装: fixest, plm, ivreg等）
# Debian bookworm標準のr-baseは4.2.2固定で、ivregの依存先（car→MatrixModels→
# Matrix>=1.6.0、R>=4.4が必要）が導入できない制約があった（Issue #171）。
# CRAN公式のDebian向けAPTリポジトリ（bookworm-cran40）を追加しR 4.x系を導入する。
# 2025年後半以降、CRANのDebian向け署名鍵配布はmarutter_pubkey.ascのURL経由ではなく
# keyserver.ubuntu.comからのGPG鍵取得方式に変更されている（Ubuntu向けとは別方式）。
# languageserverはVS CodeのR拡張機能用。
# -------------------------
RUN apt update && apt install -y --no-install-recommends \
        gnupg dirmngr ca-certificates \
    && gpg --keyserver keyserver.ubuntu.com \
        --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' \
    && gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' \
        > /etc/apt/trusted.gpg.d/cran_debian_key.asc \
    && echo "deb https://cloud.r-project.org/bin/linux/debian bookworm-cran40/" \
        > /etc/apt/sources.list.d/cran.list \
    && apt update && apt install -y --no-install-recommends \
        r-base \
        r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# ベンチマーク（benchmark/）のリファレンス実装として使うパッケージはバージョンを明示的に
# ピン留めする（Issue #239。install.packages()自体はバージョン指定に対応しないため
# remotes::install_version()を使う）。ピン留め対象外のlanguageserverはIDE補完用でベンチ
# マーク結果に影響しないため、最新版のままinstall.packages()で導入する。
# 依存関係により後段のパッケージが前段のsandwich/lmtestを別バージョンで引き込むことがある
# ため、他パッケージに依存されがちなsandwich/lmtestは最後にピン留めし直し、最終的な導入
# バージョンを確定させる。
# AER（tobit＝survival::survreg薄ラッパー、Tobit主リファレンス）とcensReg（maxLikエンジン、
# Tobit交差検証）はIssue #227で追加。両者はsandwich等を引き込むため、最終ピン留めの前に
# upgrade="never"で導入する。numDeriv はTobit限界効果のデルタ法SEをformula非依存に検証する
# （testing-policy.md「リファレンス実装」）ために追加。
RUN Rscript -e 'install.packages("remotes", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("fixest", version = "0.14.2", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("plm", version = "2.6.7", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("ivreg", version = "0.6.8", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("jsonlite", version = "2.0.0", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("marginaleffects", version = "0.32.0", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("numDeriv", version = "2016.8-1.1", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("AER", version = "1.2-15", upgrade = "never", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("censReg", version = "0.5-38", upgrade = "never", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("sandwich", version = "3.1.3", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("lmtest", version = "0.9.40", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'install.packages("languageserver", repos = "https://cloud.r-project.org")'

# 以降は一般ユーザーで実行
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# -------------------------
# vscodeユーザーが書き込めるRの個人ライブラリを用意
# (/usr/local/lib/R/site-libraryはroot所有のため、VS CodeのR拡張機能が
#  languageserver等を自動インストールしようとすると権限エラーになる。
#  R_LIBS_USERを設定しておくと、install.packages()は自動的にこちらを使う。)
# -------------------------
ENV R_LIBS_USER=/home/${USERNAME}/R/library
RUN mkdir -p ${R_LIBS_USER}

# -------------------------
# Rust (stable) インストール
# -------------------------
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup
ENV CARGO_HOME=/home/${USERNAME}/.cargo

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y \
        --default-toolchain stable \
        --component clippy \
        --component rustfmt \
        --component llvm-tools \
    && . ${CARGO_HOME}/env

ENV PATH="${CARGO_HOME}/bin:${PATH}"

RUN cargo install cargo-llvm-cov --locked

# -------------------------
# uv インストール (Python パッケージ管理)
# -------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

WORKDIR /workspaces/econometricsmodels

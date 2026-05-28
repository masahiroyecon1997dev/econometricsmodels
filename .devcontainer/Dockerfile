# ベースイメージ: Python 3.13 公式スリム (Debian bookworm)
# 軽量かつ Python 3.13 が既にインストール済み
FROM python:3.13-slim-bookworm

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=1000

# -------------------------
# システム依存パッケージ
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    # ビルドツール (Rust クレートのコンパイルに必要)
    build-essential \
    cmake \
    # パッケージリンク
    pkg-config \
    # OpenSSL (maturin, reqwest 等が依存)
    libssl-dev \
    # OpenBLAS (ndarray-linalg の BLAS バックエンド)
    libopenblas-dev \
    liblapack-dev \
    # バージョン管理
    git \
    git-lfs \
    # SSH クライアント (GitHub push)
    openssh-client \
    # CA 証明書
    ca-certificates \
    # curl (rustup / uv インストーラ)
    curl \
    # sudo (コンテナ内でパッケージ追加を可能にする)
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

# 以降は一般ユーザーで実行
USER ${USERNAME}
WORKDIR /home/${USERNAME}

# -------------------------
# Rust (stable) インストール
# -------------------------
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup
ENV CARGO_HOME=/home/${USERNAME}/.cargo

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y \
        --default-toolchain stable \
        --component clippy rustfmt llvm-tools \
    && . ${CARGO_HOME}/env

ENV PATH="${CARGO_HOME}/bin:${PATH}"

# カバレッジ計測ツール
RUN cargo install cargo-llvm-cov --locked

# -------------------------
# uv インストール (Python パッケージ管理)
# -------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

# -------------------------
# ワークスペースをデフォルト作業ディレクトリに設定
# -------------------------
WORKDIR /workspaces/econometricsmodels

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
    libopenblas-dev \
    liblapack-dev \
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
# Node.js (Node 24系) インストール & Claude Codeインストール
# -------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt install -y nodejs \
    && npm install -g @anthropic-ai/claude-code \
    && rm -rf /var/lib/apt/lists/*

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

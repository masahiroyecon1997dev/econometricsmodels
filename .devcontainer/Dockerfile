# uvのバイナリ取得元。Dependabot(docker)は`FROM`行しか解析せず、`COPY --from=<image>`に
# 直接書いた参照は追跡されないため、`FROM ... AS uv`のステージとして宣言し、
# 下の`COPY --from=uv`で参照する。版とダイジェストの更新はDependabot(docker)が行う。
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv

# ベースイメージはタグだけでなくダイジェストでも固定する(同じタグの中身が差し替わっても
# 検知できる)。ダイジェストの更新はDependabot(docker)が行う。
FROM python:3.14-slim-bookworm@sha256:945c2676dc2b18e0c1f16879244122c57c0ffede60006bf3df159c507a31bcae

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
    ca-certificates \
    curl \
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
        ${USERNAME}
# sudoは入れない(ビルド後にパッケージを足す必要があればこのDockerfileに書いて再構築する)。
# SSH(openssh-client・ホスト鍵の事前登録)も入れない(リモートはHTTPSで、git pushはVS Codeの
# 認証情報転送を使う)。

# -------------------------
# Node.js (Node 24系) インストール
# Claude Code CLI自体はdevcontainer.jsonのfeature（ghcr.io/anthropics/devcontainer-features/claude-code）
# が入れるため、ここでは重複させずNode.jsのみ導入する。Dockerfileで先に入れておくのは、
# featureが自前で入れるNode 18(EOL)を避けるため。
# NodeSourceのsetupスクリプトをrootで`curl | bash`せず、APTリポジトリを署名鍵
# (`signed-by`)つきで追加する。鍵のフィンガープリントはNodeSource公式の鍵のもの。
# -------------------------
ARG NODESOURCE_KEY_FPR=6F71F525282841EEDAF851B42F59B5F99B1BE0B4
RUN apt update && apt install -y --no-install-recommends gnupg \
    && install -d -m 0755 /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key -o /tmp/nodesource.key \
    && gpg --show-keys --with-colons /tmp/nodesource.key | grep -q "^fpr:::::::::${NODESOURCE_KEY_FPR}:" \
    && gpg --dearmor < /tmp/nodesource.key > /etc/apt/keyrings/nodesource.gpg \
    && rm /tmp/nodesource.key \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_24.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt update && apt install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# R インストール（ベンチマーク用リファレンス実装: fixest, plm, ivreg等）
# Debian bookworm標準のr-baseは4.2.2固定で、ivregの依存先（car→MatrixModels→
# Matrix>=1.6.0、R>=4.4が必要）が導入できない制約があった。
# CRAN公式のDebian向けAPTリポジトリ（bookworm-cran40）を追加しR 4.x系を導入する。
# 2025年後半以降、CRANのDebian向け署名鍵配布はmarutter_pubkey.ascのURL経由ではなく
# keyserver.ubuntu.comからのGPG鍵取得方式に変更されている（Ubuntu向けとは別方式）。
# languageserverはVS CodeのR拡張機能用。
# -------------------------
# 鍵は/etc/apt/trusted.gpg.d/ではなく/etc/apt/keyrings/に置き、`signed-by`でCRANの
# リポジトリだけに使う(trusted.gpg.d/に置くと全てのAPTリポジトリの署名検証に使われる)。
RUN apt update && apt install -y --no-install-recommends \
        gnupg dirmngr ca-certificates \
    && install -d -m 0755 /etc/apt/keyrings \
    && gpg --keyserver keyserver.ubuntu.com \
        --recv-key '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' \
    && gpg --armor --export '95C0FAF38DB3CCAD0C080A7BDC78B2DDEABC47B7' \
        > /etc/apt/keyrings/cran_debian_key.asc \
    && echo "deb [signed-by=/etc/apt/keyrings/cran_debian_key.asc] https://cloud.r-project.org/bin/linux/debian bookworm-cran40/" \
        > /etc/apt/sources.list.d/cran.list \
    && apt update && apt install -y --no-install-recommends \
        r-base \
        r-base-dev \
    && rm -rf /var/lib/apt/lists/*

# ベンチマーク（benchmark/）のリファレンス実装として使うパッケージはバージョンを明示的に
# ピン留めする（install.packages()自体はバージョン指定に対応しないため
# remotes::install_version()を使う）。ピン留め対象外のlanguageserverはIDE補完用でベンチ
# マーク結果に影響しないため、最新版のままinstall.packages()で導入する。
# ピン留めバージョンの更新は.github/workflows/check_r_updates.ymlが週次で確認し、Issueで通知する。
# 依存関係により後段のパッケージが前段のsandwich/lmtestを別バージョンで引き込むことがある
# ため、他パッケージに依存されがちなsandwich/lmtestは最後にピン留めし直し、最終的な導入
# バージョンを確定させる。
# AER（tobit＝survival::survreg薄ラッパー、Tobit主リファレンス）とcensReg（maxLikエンジン、
# Tobit交差検証）を追加。両者はsandwich等を引き込むため、最終ピン留めの前に
# upgrade="never"で導入する。numDeriv はTobit限界効果のデルタ法SEをformula非依存に検証する
# （testing-policy.md「リファレンス実装」）ために追加。
# momentfit はIV-GMMの独立実装によるクロスチェック（ivregはGMM非対応）。sandwichに依存する
# ため、最終ピン留めの前にupgrade="never"で導入する。
# maxLik（censRegの依存先、Tobit交差検証の最適化エンジン）は、censRegが依存として引き込む
# バージョンが導入時期で変わる（フィクスチャ記録では1.5.2.2→1.6.10）ため、censRegより前に
# 明示的にピン留めする。
RUN Rscript -e 'install.packages("remotes", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("fixest", version = "0.14.2", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("plm", version = "2.6.7", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("ivreg", version = "0.6.8", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("jsonlite", version = "2.0.0", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("marginaleffects", version = "1.0.0", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("numDeriv", version = "2016.8-1.1", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("maxLik", version = "1.6-10", upgrade = "never", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("AER", version = "1.2-17", upgrade = "never", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("censReg", version = "0.5-40", upgrade = "never", repos = "https://cloud.r-project.org")' \
    && Rscript -e 'remotes::install_version("momentfit", version = "1.0", upgrade = "never", repos = "https://cloud.r-project.org")' \
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
# Rust インストール
# ツールチェーンのバージョンを固定する(CIは`setup-rust-toolchain`でstableを使うため、
# 数バージョン先で差が出うる。上げるときはここだけ手動で上げる)。rustupのブートストラップ
# 自体は公式のインストーラを`curl | sh`で実行する(HTTPS・TLS1.2以上)。
# -------------------------
ARG RUST_VERSION=1.99.0
ENV RUSTUP_HOME=/home/${USERNAME}/.rustup
ENV CARGO_HOME=/home/${USERNAME}/.cargo

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y \
        --default-toolchain ${RUST_VERSION} \
        --component clippy \
        --component rustfmt \
        --component llvm-tools \
    && . ${CARGO_HOME}/env

ENV PATH="${CARGO_HOME}/bin:${PATH}"

# サプライチェーン攻撃対策として、cargo経由で入れる開発ツールはバージョンを固定し、
# ソースからビルドする（`--locked`でツール自身の依存もそのクレートのCargo.lockで固定）。
# cargo-auditのバージョンはCIの`ci_engine.yml`（`tool: cargo-audit@...`）と揃えること。
# 新版の有無と両者の食い違いは.github/workflows/check_cargo_tool_updates.ymlが週次で確認し、
# Issueで通知する（`ARG`名と`cargo install`行の`<crate>@${ARG}`の書式を変えると解析できなくなる）。
ARG CARGO_AUDIT_VERSION=0.22.2
ARG CARGO_ABOUT_VERSION=0.9.2
ARG CARGO_LLVM_COV_VERSION=0.9.1

RUN cargo install --locked \
        cargo-audit@${CARGO_AUDIT_VERSION} \
        cargo-llvm-cov@${CARGO_LLVM_COV_VERSION}

# cargo-aboutはdefault featureを持たず、バイナリは`cli`フィーチャーが必須
# （`required-features = ["cli"]`）。`--features cli`を付けないとバイナリが無いまま
# 警告だけでインストールが終わるため、上の行とは分けて`--features cli`を付ける
# （まとめると、featureを持たない他のcrateにも`--features cli`が掛かってしまう）。
RUN cargo install --locked cargo-about@${CARGO_ABOUT_VERSION} --features cli

# 必要なサブコマンドが実際に使えることをビルド時に確認し、欠けていればイメージのビルドを失敗させる
# （リリース準備で`cargo about generate`等を実行するまで気づかない事態を防ぐ）。
RUN cargo audit --version \
    && cargo about --version \
    && cargo llvm-cov --version

# -------------------------
# uv (Python パッケージ管理)
# `curl | sh`ではなく、版とダイジェストを固定した公式イメージからバイナリをコピーする
# (`uvx`は使わない方針だが、uvのイメージには同梱されるため一緒にコピーされる)。
# バージョンとダイジェストはファイル先頭の`FROM ... AS uv`で固定している
# (更新はDependabot(docker)が行う)。
# -------------------------
COPY --from=uv /uv /uvx /usr/local/bin/

ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"

WORKDIR /workspaces/econometricsmodels

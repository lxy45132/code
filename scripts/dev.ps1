# 本地开发一条命令化（docs/03 §10）。
# 用法：.\scripts\dev.ps1 api|worker|web|migrate|test|build-frontend|e2e
# 前置：uv sync --extra dev --extra train --extra web；Redis 已启动；
#       MySQL 模式需先 .\scripts\dev.ps1 migrate（或 db init）。
param(
    [Parameter(Position = 0)]
    [ValidateSet("api", "worker", "web", "migrate", "test", "build-frontend", "e2e")]
    [string]$Task = "api"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

switch ($Task) {
    "api" { uv run --no-sync hybrid-inpaint start -c configs/system.yaml }
    "worker" { uv run --no-sync hybrid-inpaint worker -c configs/system.yaml }
    "web" { npm --prefix frontend run dev }
    "migrate" { uv run --no-sync hybrid-inpaint db init -c configs/system.yaml }
    "test" {
        uv run --no-sync pytest
        npm --prefix frontend run test
    }
    "build-frontend" { npm --prefix frontend run build }
    "e2e" { npm --prefix frontend run test:e2e }
}

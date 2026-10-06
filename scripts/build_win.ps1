# Windows 原生构建 lbrr.exe + cudnn_sdpa_wrap.dll（一条龙）
# 前置：MSVC Build Tools（cl 在 PATH 或 VCINSTALLDIR 可寻）、CUDA Toolkit
# 12.x/13.x、LLVM/libclang、rustup stable。运行库（cuDNN DLL、cfe 头文件）
# 若缺失会给出布置提示。
param(
    [string]$CudaVersion = "v13.4",
    [string]$LibsDir = "D:\Projects\lbrr-win-libs",
    [switch]$SkipShim
)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
Set-Location $repo

# --- 依赖落盘约定：Rust 工具链/缓存一律 D 盘，禁止写 C 盘 ---
if (Test-Path "D:\cargo\bin")   { $env:CARGO_HOME  = "D:\cargo"; $env:PATH = "D:\cargo\bin;$env:PATH" }
if (Test-Path "D:\rustup")      { $env:RUSTUP_HOME = "D:\rustup" }
if ($env:CARGO_HOME -like "C:*" -or $env:RUSTUP_HOME -like "C:*") {
    throw "CARGO_HOME/RUSTUP_HOME 指向 C 盘（$env:CARGO_HOME / $env:RUSTUP_HOME），违反依赖不落 C 盘的约定"
}

# --- CUDA 环境 ---
$cudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$CudaVersion"
if (-not (Test-Path $cudaHome)) { throw "CUDA Toolkit not found: $cudaHome" }
$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$env:CUDA_TOOLKIT_PATH = $cudaHome
if ($env:LIBCLANG_PATH -eq $null) { $env:LIBCLANG_PATH = "D:\Softwares\LLVM\bin" }
$env:PATH = "$cudaHome\bin;$env:LIBCLANG_PATH;$env:PATH"

# --- cudnn 运行库布局检查 ---
$cudnnBin = Join-Path $LibsDir "cudnn\bin"
$cfeInclude = Join-Path $LibsDir "cfe\cudnn-frontend-v1212\include"
if (-not (Test-Path (Join-Path $cudnnBin "cudnn64_9.dll"))) {
    Write-Host @"
缺少 cuDNN DLL: $cudnnBin
布置方法（一次性）:
  Invoke-RestMethod https://pypi.org/pypi/nvidia-cudnn-cu13/json 查最新 win_amd64 wheel
  下载 -> Expand-Archive -> nvidia/cudnn/bin 与 include 拷到 $LibsDir\cudnn\
"@
    throw "cudnn runtime not prepared"
}
if (-not (Test-Path $cfeInclude)) {
    Write-Host "缺少 cudnn-frontend 头文件: $cfeInclude（WSL /opt/lbrr-cudnn/cfe 拷贝）"
    throw "cfe headers not prepared"
}

# --- 定位 MSVC（统一经 vcvars：cl 在 PATH 不代表 INCLUDE/LIB 环境就绪）---
if (-not $SkipShim) {
    $vcvars = Get-ChildItem "$env:VCINSTALLDIR\Auxiliary\Build\vcvars64.bat",
        "F:\vs\VC\Auxiliary\Build\vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($vcvars) {
        Write-Host "== cl 不在 PATH，经 vcvars 构建环境编译 shim =="
        $shimOut = Join-Path $LibsDir "cudnn_sdpa_wrap.dll"
        $shortInclude = (New-Object -ComObject Scripting.FileSystemObject).GetFolder("$cudaHome\include").ShortPath
        # vcvars 内的 cl 必须赢过 PATH 上可能存在的 LLVM clang-cl：
        # 临时摘掉 LLVM 目录再起子进程。
        $savedPath = $env:PATH
        $env:PATH = (($env:PATH -split ';') | Where-Object { $_ -and $_ -ne $env:LIBCLANG_PATH }) -join ';'
        cmd /c "`"$($vcvars.FullName)`" >nul 2>&1 && cl /LD /O2 /EHsc /std:c++17 /utf-8 /DWIN32_LEAN_AND_MEAN /DNV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING=1 -I$repo\tools\cfe-win-override -I$cfeInclude -I$cudnnBin\..\include -I$shortInclude $repo\tools\cudnn_sdpa_wrap.cpp /Fe:$shimOut"
        $env:PATH = $savedPath
        if ($LASTEXITCODE -ne 0) { throw "shim compile failed" }
        Write-Host "shim -> $shimOut"
    } else {
        Write-Warning "未找到 vcvars64.bat，跳过 shim 编译（SDPA 将回退手写 kernel）"
    }
}

# --- lbrr.exe ---
Write-Host "== cargo oxide build --release =="
cargo oxide build -- --release
if ($LASTEXITCODE -ne 0) { throw "lbrr build failed" }
Write-Host ""
Write-Host "lbrr.exe -> $repo\target\release\lbrr.exe"
Write-Host "运行: powershell $repo\scripts\lbrr.ps1 --separate --input <audio> --outdir <dir>"

# lbrr Windows 运行入口：补齐运行期环境变量后转发参数给 lbrr.exe
# 用法示例:
#   powershell scripts\lbrr.ps1 --separate --input song.wav --outdir out
#   powershell scripts\lbrr.ps1 --bench --iters 5 --model-dir assets
param(
    [string]$CudaVersion = "v13.4",
    [string]$LibsDir = "D:\Projects\lbrr-win-libs",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$LbrrArgs
)
$repo = Split-Path $PSScriptRoot -Parent
$exe = Join-Path $repo "target\release\lbrr.exe"
if (-not (Test-Path $exe)) { throw "lbrr.exe 不存在，先运行 scripts\build_win.ps1" }

# cudnn.rs / cublaslt.rs 通过 CUDA_HOME 推导 toolkit bin；默认注入。
$cudaHome = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$CudaVersion"
if (Test-Path $cudaHome) {
    if (-not $env:CUDA_HOME) { $env:CUDA_HOME = $cudaHome }
    if (-not $env:CUDA_PATH) { $env:CUDA_PATH = $cudaHome }
}
# 运行库默认值即 D:/Projects/lbrr-win-libs（源码内建），此处只做存在性提示。
if (-not (Test-Path (Join-Path $LibsDir "cudnn_sdpa_wrap.dll"))) {
    Write-Warning "未找到 $LibsDir\cudnn_sdpa_wrap.dll，SDPA 将回退手写 kernel（较慢）"
}

& $exe @LbrrArgs
exit $LASTEXITCODE

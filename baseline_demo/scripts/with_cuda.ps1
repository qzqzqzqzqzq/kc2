param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArgs)
$ErrorActionPreference = 'Stop'
$vs = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
# Import the compiler environment only into this task process.
$compilerEnv = & cmd.exe /s /c "`"$vs`" >nul && set"
foreach ($line in $compilerEnv) {
    if ($line -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1],$matches[2],'Process')
    }
}
$env:CUDA_HOME = 'D:\vggt_cuda128'
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "D:\vggt311\Scripts;$env:CUDA_HOME\bin;$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = '8.9'
$env:MAX_JOBS = '4'
$env:DISTUTILS_USE_SDK = '1'
$env:PYTHONUTF8 = '1'
& D:\vggt311\Scripts\python.exe @PythonArgs
exit $LASTEXITCODE

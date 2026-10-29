[CmdletBinding()]
param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'build'),
    [switch]$PadToOneGB
)
$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path -LiteralPath $compiler)) { throw 'Install the Windows .NET Framework x64 compiler.' }
$destination = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$executable = Join-Path $destination 'DM3D_JARVIS_X.exe'
& $compiler /nologo /optimize+ /platform:x64 /target:exe /r:System.Numerics.dll "/out:$executable" (Join-Path $PSScriptRoot 'DM3D.cs')
if ($LASTEXITCODE -ne 0) { throw 'Compilation failed' }
& $executable --self-test
if ($LASTEXITCODE -ne 0) { throw 'Compact executable self-test failed' }
$imageBytes = (Get-Item -LiteralPath $executable).Length
if ($PadToOneGB) {
    if ($imageBytes -gt 1000000000) { throw 'Executable exceeds distribution budget' }
    $stream = [IO.File]::Open($executable, 'Open', 'Write', 'None')
    try { $stream.SetLength(1000000000); $stream.Flush($true) } finally { $stream.Dispose() }
    & $executable --self-test
    if ($LASTEXITCODE -ne 0) { throw 'Padded executable self-test failed' }
}
$data = Join-Path $destination 'DM3D_Data'
New-Item -ItemType Directory -Force -Path $data | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'examples\demo.asm') -Destination (Join-Path $data 'demo.asm')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'examples\demo.txt') -Destination (Join-Path $destination 'DM3D_Demo.txt')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination (Join-Path $destination 'README.md')
$checksum = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
$length = (Get-Item -LiteralPath $executable).Length
[IO.File]::WriteAllText((Join-Path $destination 'DM3D_JARVIS_X.sha256'), "$checksum  DM3D_JARVIS_X.exe" + [Environment]::NewLine)
$manifest = [ordered]@{
    format = 'dm3d-windows-build-v1'
    source_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'DM3D.cs') -Algorithm SHA256).Hash.ToLowerInvariant()
    executable_sha256 = $checksum
    executable_bytes = $length
    image_bytes = $imageBytes
    reserved_bytes = $length - $imageBytes
    padded_to_decimal_gb = [bool]$PadToOneGB
    runtime = '.NET Framework x64'
}
[IO.File]::WriteAllText((Join-Path $destination 'build-manifest.json'), ($manifest | ConvertTo-Json) + [Environment]::NewLine)
Write-Output "Built $length bytes; image $imageBytes bytes; SHA256 $checksum"

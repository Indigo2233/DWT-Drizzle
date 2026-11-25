param(
    [Parameter(Mandatory = $true)]
    [string] $PixInsightScripts,
    [string] $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
)

$ErrorActionPreference = "Stop"
$source = Join-Path $RepositoryRoot "core-rs\wbpp"
$destination = Join-Path (Resolve-Path $PixInsightScripts).Path "DWT-Drizzle"

New-Item -ItemType Directory -Path $destination -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $source "pipeline-builder.js") -Destination $destination -Force
Copy-Item -LiteralPath (Join-Path $source "pipeline-builder-ui.js") -Destination $destination -Force
Copy-Item -LiteralPath (Join-Path $source "event-script.js") -Destination $destination -Force
Copy-Item -LiteralPath (Join-Path $source "README.md") -Destination $destination -Force

$binary = Join-Path $source "bin/dwt-backend.exe"
if (Test-Path -LiteralPath $binary)
{
    $binaryDestination = Join-Path $destination "bin"
    New-Item -ItemType Directory -Path $binaryDestination -Force | Out-Null
    Copy-Item -LiteralPath $binary -Destination $binaryDestination -Force
}

$binary = Join-Path $RepositoryRoot "core-rs\target\release\dwt-backend.exe"
if (Test-Path -LiteralPath $binary)
{
    $binaryDestination = Join-Path $destination "bin"
    New-Item -ItemType Directory -Path $binaryDestination -Force | Out-Null
    Copy-Item -LiteralPath $binary -Destination $binaryDestination -Force
}
else
{
    Write-Warning "Build the Rust backend first: cargo build --manifest-path core-rs/Cargo.toml --release --bin dwt-backend"
}

$configPath = Join-Path $destination "dwt-backend.json"
if (!(Test-Path -LiteralPath $configPath))
{
    $config = @{
        executable = (Join-Path $destination "bin/dwt-backend.exe").Replace("\", "/")
    }
    $config | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $configPath -Encoding utf8
}

Write-Output "Installed DWT-Drizzle WBPP scripts in $destination"
Write-Output "Interactive Pipeline Builder: $(Join-Path $destination 'pipeline-builder-ui.js')"
Write-Output "Unattended Pipeline Builder:  $(Join-Path $destination 'pipeline-builder.js')"
Write-Output "Event script:                 $(Join-Path $destination 'event-script.js')"

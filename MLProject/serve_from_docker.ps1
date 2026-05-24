param(
    [string]$ImageName = 'profdara/mlproject-model:latest',
    [int]$HostPort = 5001
)

$ErrorActionPreference = 'Stop'

Write-Host "Pulling image: $ImageName"
docker pull $ImageName

Write-Host "Starting model server on http://localhost:$HostPort"
docker run --rm -p "$HostPort`:8080" $ImageName
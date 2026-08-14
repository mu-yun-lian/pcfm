$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$gateScript = (
    "C:\Users\14013\.codex\skills\" +
    "build-falsifiable-cognitive-modules\scripts\module_gate.py"
)
$manifest = Join-Path $projectRoot "MODULE_GATE_DECISION_EVIDENCE_V1.json"

Push-Location $projectRoot
try {
    $gateOutput = & python $gateScript check $manifest --repo-root . 2>&1
    $gateExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

$gateText = $gateOutput -join "`n"
$expected = "completion gate requires module.status implemented_confirmatory"
if ($gateExitCode -ne 1 -or $gateText -notmatch [regex]::Escape($expected)) {
    Write-Error $gateText
    exit 1
}

Write-Output (
    "Expected completion refusal confirmed: " +
    "implemented_unintegrated is not confirmatory."
)


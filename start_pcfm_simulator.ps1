$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = Join-Path $projectRoot "src"
$dataDir = Join-Path $projectRoot "artifacts\conversation_mvp_v03\local_runtime"
Set-Location -LiteralPath $projectRoot

try {
    python -m pcfm.webapp --data-dir $dataDir --seed-demos
}
catch {
    Write-Host "PCFM 人物对话系统启动失败：$($_.Exception.Message)" -ForegroundColor Red
    Read-Host "按回车键关闭"
}

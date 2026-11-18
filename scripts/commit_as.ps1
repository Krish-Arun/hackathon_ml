# PowerShell equivalent for committing from Windows.
#   .\scripts\commit_as.ps1 krish "feat: candidate generation"
param([Parameter(Mandatory=$true)][string]$Who,
      [Parameter(Mandatory=$true, ValueFromRemainingArguments=$true)][string[]]$Message)

$people = @{
  krish    = @{ Name = "Krish Arun";     Email = "krish.official.v1@gmail.com" }
  niharika = @{ Name = "Niharika Mirle"; Email = "neha.mirle17@gmail.com" }
  mrudul   = @{ Name = "Mrudul";         Email = "chinju1206@gmail.com" }
}
if (-not $people.ContainsKey($Who)) { Write-Error "usage: commit_as.ps1 {krish|niharika|mrudul} <message>"; exit 1 }

$p = $people[$Who]
$env:GIT_AUTHOR_NAME = $p.Name;    $env:GIT_AUTHOR_EMAIL = $p.Email
$env:GIT_COMMITTER_NAME = $p.Name; $env:GIT_COMMITTER_EMAIL = $p.Email
git commit -m ($Message -join " ")
Write-Host "committed as $($p.Name) <$($p.Email)>"

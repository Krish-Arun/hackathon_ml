# Commit as Krish with optional co-author trailers, from Windows.
#   .\scripts\commit.ps1 "feat: feature engineering" niharika mrudul
param([Parameter(Mandatory=$true)][string]$Message,
      [Parameter(ValueFromRemainingArguments=$true)][string[]]$CoAuthors)

$trailers = @{
  niharika = "Co-authored-by: Niharika Mirle <neha.mirle17@gmail.com>"
  mrudul   = "Co-authored-by: Mrudul <chinju1206@gmail.com>"
}

$lines = @($Message)
if ($CoAuthors) {
  $lines += ""
  foreach ($c in $CoAuthors) {
    if ($trailers.ContainsKey($c)) { $lines += $trailers[$c] }
    else { Write-Error "unknown teammate: $c"; exit 1 }
  }
}

$env:GIT_AUTHOR_NAME = "Krish Arun";    $env:GIT_AUTHOR_EMAIL = "krish.official.v1@gmail.com"
$env:GIT_COMMITTER_NAME = "Krish Arun"; $env:GIT_COMMITTER_EMAIL = "krish.official.v1@gmail.com"
git commit -m ($lines -join "`n")

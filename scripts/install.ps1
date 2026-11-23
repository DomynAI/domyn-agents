# domyn installer for Windows (x64, and ARM64 through its x64 emulation).
#
#   irm https://github.com/domynai/domyn-agents/releases/latest/download/install.ps1 | iex
#   & .\install.ps1 <dir>      install from a local build directory instead
#
# The counterpart of install.sh: the same archive layout, checks and environment
# variables, in Windows PowerShell 5.1 or PowerShell 7. Downloads from the GitHub
# releases unless a base URL is given.
#
# Env:
#   DOMYN_VERSION       a release to install, e.g. 1.4.26 (default: the latest)
#   DOMYN_BASE_URL      a folder holding the archives (e.g. a self-hosted mirror);
#                       used as is, so it picks the version itself
#   DOMYN_RELEASES_URL  another repository's releases (a fork)
#   DOMYN_HOME          runtime location (default %LOCALAPPDATA%\Programs\domyn)
#   DOMYN_BIN_DIR       where the launcher goes (default %USERPROFILE%\.local\bin)
#   DOMYN_LOCAL_DIR     local build directory, same as the first argument
#
# Run through `iex` it shares the caller's session: nothing here may `exit` (that
# closes the window) or change the caller's preferences, so everything happens
# inside one function and a failure is thrown and reported, not exited on.

$ReleasesUrl = 'https://github.com/domynai/domyn-agents/releases'
# Set by `INSTALL_BASE_URL=... scripts/build_standalone.sh` in the copy it writes.
$BaseUrl = ""

function Install-Domyn {
    param([string] $LocalDir)

    $ErrorActionPreference = 'Stop'
    # Windows PowerShell 5.1 draws a progress bar for every block downloaded, which
    # makes a 100 MB download several times slower.
    $ProgressPreference = 'SilentlyContinue'

    function Write-Info([string] $Text) { Write-Host '==> ' -ForegroundColor Cyan -NoNewline; Write-Host $Text }
    function Write-Ok([string] $Text) { Write-Host ' ok ' -ForegroundColor Green -NoNewline; Write-Host $Text }
    function Write-Warn([string] $Text) { Write-Host '  ! ' -ForegroundColor Yellow -NoNewline; Write-Host $Text }
    function Stop-Install([string] $Text) { throw [System.InvalidOperationException]::new($Text) }

    # A folder just made or just filled can be held for a moment by an antivirus
    # scanning it: seen once as "access denied" making the unpacking folder.
    function Invoke-Retried([scriptblock] $Action) {
        for ($try = 1; ; $try++) {
            try { return & $Action } catch {
                if ($try -ge 5) { throw }
                Start-Sleep -Milliseconds 400
            }
        }
    }

    function Get-Sha256([string] $Path) {
        (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
    }

    # A native program's stderr, redirected, is an error record in 5.1, which
    # 'Stop' turns into an exception even when the program succeeded: the exit
    # code is what decides.
    function Invoke-Native([string] $Program, [string[]] $Arguments) {
        $saved = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $out = & $Program @Arguments 2>&1 | ForEach-Object { "$_" }
            return [pscustomobject]@{ Code = $LASTEXITCODE; Output = @($out) }
        } finally {
            $ErrorActionPreference = $saved
        }
    }

    if (-not $LocalDir) { $LocalDir = $env:DOMYN_LOCAL_DIR }
    $base = if ($env:DOMYN_BASE_URL) { $env:DOMYN_BASE_URL } else { $BaseUrl }
    $releases = if ($env:DOMYN_RELEASES_URL) { $env:DOMYN_RELEASES_URL } else { $ReleasesUrl }
    $version = "$env:DOMYN_VERSION".TrimStart('v')
    $homeDir = if ($env:DOMYN_HOME) { $env:DOMYN_HOME } else { Join-Path $env:LOCALAPPDATA 'Programs\domyn' }
    $binDir = if ($env:DOMYN_BIN_DIR) { $env:DOMYN_BIN_DIR } else { Join-Path $env:USERPROFILE '.local\bin' }
    $homeDir = [System.IO.Path]::GetFullPath($homeDir)
    $binDir = [System.IO.Path]::GetFullPath($binDir)
    $incoming = "$homeDir.incoming"
    $previous = "$homeDir.previous"
    $marker = '.domyn-runtime'

    Write-Host ''
    Write-Host '  domyn' -ForegroundColor Cyan -NoNewline
    Write-Host '  installer' -ForegroundColor DarkGray
    Write-Host ''

    if ($LocalDir) {
        if (-not (Test-Path -LiteralPath $LocalDir -PathType Container)) { Stop-Install "no such directory: $LocalDir" }
        $LocalDir = (Resolve-Path -LiteralPath $LocalDir).Path
        Write-Info "installing from local build: $LocalDir"
    } elseif ($base -and $version) {
        Stop-Install 'DOMYN_VERSION applies to the GitHub releases only; with a base URL, point it at that version''s folder instead'
    }

    # Windows' own tar (bsdtar, Windows 10 1803 and later): an order of magnitude
    # faster than Expand-Archive on the runtime's thousands of files.
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (-not (Test-Path -LiteralPath $tar)) { Stop-Install 'tar.exe is required (Windows 10 1803 or later)' }

    # The runtime is replaced wholesale: never replace a directory we did not make.
    # Ours carries the marker, so a damaged install can still be repaired.
    if ((Test-Path -LiteralPath $homeDir) -and (Get-ChildItem -LiteralPath $homeDir -Force | Select-Object -First 1) `
            -and -not (Test-Path -LiteralPath (Join-Path $homeDir $marker)) `
            -and -not (Test-Path -LiteralPath (Join-Path $homeDir 'python\python.exe'))) {
        Stop-Install "$homeDir exists and is not a domyn runtime; refusing to replace it (set DOMYN_HOME elsewhere)"
    }

    # x64 everywhere: Windows on ARM runs it emulated, and some of domyn's
    # dependencies publish no ARM64 build for Windows.
    $arch = $env:PROCESSOR_ARCHITEW6432
    if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
    switch ($arch) {
        'AMD64' { }
        'ARM64' { Write-Info 'Windows on ARM: installing the x64 build, which runs emulated' }
        default { Stop-Install "unsupported processor: $arch (domyn needs 64-bit Windows)" }
    }
    $target = 'x86_64-pc-windows-msvc'
    $archive = "domyn-$target.tar.gz"
    if ($base) { $from = $base.TrimEnd('/') }
    elseif ($version) { $from = "$($releases.TrimEnd('/'))/download/v$version" }
    else { $from = "$($releases.TrimEnd('/'))/latest/download" }
    $url = "$from/$archive"

    $work = Join-Path ([System.IO.Path]::GetTempPath()) ("domyn-install-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $work | Out-Null
    $swapped = $false
    try {
        $archivePath = Join-Path $work $archive
        $sumPath = "$archivePath.sha256"
        if ($LocalDir) {
            $local = Join-Path $LocalDir $archive
            if (-not (Test-Path -LiteralPath $local)) { Stop-Install "not found: $local (built for a different platform?)" }
            Write-Info "copying domyn for $target"
            Copy-Item -LiteralPath $local -Destination $archivePath
            if (Test-Path -LiteralPath "$local.sha256") { Copy-Item -LiteralPath "$local.sha256" -Destination $sumPath }
        } else {
            # TLS 1.2 is not on by default in Windows PowerShell 5.1 on older systems.
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
            $label = if ($version) { " $version" } else { '' }
            Write-Info "downloading domyn$label for $target from $from"
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $archivePath
            } catch {
                $hint = if ($version) { " (is there a release v$version?)" } else { '' }
                Stop-Install "download failed: $url$hint"
            }
            try {
                Invoke-WebRequest -UseBasicParsing -Uri "$url.sha256" -OutFile $sumPath
            } catch {
                Stop-Install "could not download the checksum: $url.sha256"
            }
        }

        if (Test-Path -LiteralPath $sumPath) {
            Write-Info 'verifying checksum'
            $expected = ((Get-Content -LiteralPath $sumPath -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
            if ($expected -ne (Get-Sha256 $archivePath)) { Stop-Install 'checksum mismatch - the download is not what was published' }
            Write-Ok 'checksum matches'
        } else {
            Write-Warn 'no checksum next to the local build; installing unverified'
        }

        Write-Info 'unpacking'
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $homeDir) | Out-Null
        if (Test-Path -LiteralPath $incoming) { Invoke-Retried { Remove-Item -LiteralPath $incoming -Recurse -Force } }
        Invoke-Retried { New-Item -ItemType Directory -Force -Path $incoming | Out-Null }
        $unpack = Invoke-Native $tar @('-xzf', $archivePath, '-C', $incoming)
        if ($unpack.Code -ne 0) { Stop-Install "could not unpack $archive`n$($unpack.Output -join "`n")" }
        $python = Join-Path $incoming 'python\python.exe'
        if (-not (Test-Path -LiteralPath $python)) { Stop-Install 'the archive does not contain a runtime' }

        # Checked before the swap, so a build that cannot run here leaves the
        # previous install untouched. Python's code goes in a file: Windows
        # PowerShell 5.1 passes a double quote inside an argument unescaped.
        $check = Join-Path $work 'check.py'
        Set-Content -LiteralPath $check -Encoding ASCII -Value @'
import importlib, importlib.util
import domyn.agents.cli.main
from domyn.agents.cli.utils import get_version
for name in ("pydantic_core", "yaml", "cryptography.hazmat.bindings._rust", "rpds", "anydoc", "watchfiles", "pdf_inspector"):
    if importlib.util.find_spec(name.split(".")[0]) is not None:
        importlib.import_module(name)
print(get_version())
'@
        $result = Invoke-Native $python @('-P', $check)
        if ($result.Code -ne 0) {
            $result.Output | Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
            Stop-Install 'the new runtime does not start on this machine; nothing was changed'
        }
        $installed = $result.Output | Select-Object -Last 1
        Set-Content -LiteralPath (Join-Path $incoming $marker) -Value "domyn $installed, installed by install.ps1" -Encoding ASCII

        # A file in use cannot be moved on Windows: a domyn still running from the
        # old runtime keeps it in place, and the new one is not half-installed.
        if (Test-Path -LiteralPath $previous) { Invoke-Retried { Remove-Item -LiteralPath $previous -Recurse -Force } }
        if (Test-Path -LiteralPath $homeDir) {
            try {
                Invoke-Retried { Rename-Item -LiteralPath $homeDir -NewName (Split-Path -Leaf $previous) }
            } catch {
                Stop-Install "could not replace $homeDir, which is in use: close every domyn session and run the installer again"
            }
        }
        Invoke-Retried { Rename-Item -LiteralPath $incoming -NewName (Split-Path -Leaf $homeDir) }
        $swapped = $true
        if (Test-Path -LiteralPath $previous) {
            Remove-Item -LiteralPath $previous -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Ok "runtime installed in $homeDir"
        $python = Join-Path $homeDir 'python\python.exe'

        # A config embedded in the build is copied only where the user has none.
        $seed = Join-Path $homeDir 'seed'
        $planted = @()
        $config = Join-Path $env:USERPROFILE '.domyn'
        if (Test-Path -LiteralPath $seed) {
            New-Item -ItemType Directory -Force -Path $config | Out-Null
            foreach ($f in 'config.yaml', 'credentials') {
                $source = Join-Path $seed $f
                if (-not (Test-Path -LiteralPath $source)) { continue }
                if (Test-Path -LiteralPath (Join-Path $config $f)) {
                    Write-Info "keeping your existing $f"
                } else {
                    Copy-Item -LiteralPath $source -Destination (Join-Path $config $f)
                    $planted += $f
                }
            }
            if ($planted) { Write-Ok "configuration installed ($($planted -join ' '))" }
        }

        # domyn.exe, made as pip makes a command's launcher (distlib, from the pip
        # in the runtime), pointing at this runtime. An .exe and not a .cmd: Ctrl-C
        # in a batch file asks "Terminate batch job (Y/N)?" after domyn has handled
        # it. The launcher's own script keeps the current folder off sys.path.
        New-Item -ItemType Directory -Force -Path $binDir | Out-Null
        $launcher = Join-Path $binDir 'domyn.exe'
        $make = Join-Path $work 'launcher.py'
        Set-Content -LiteralPath $make -Encoding ASCII -Value @'
import sys
from pip._vendor.distlib.scripts import ScriptMaker
maker = ScriptMaker(None, sys.argv[1], add_launchers=True)
maker.executable = sys.argv[2]
maker.variants = {""}
maker.clobber = True
maker.make("domyn = domyn.agents.cli.main:cli")
'@
        $made = Invoke-Native $python @('-P', $make, $binDir, $python)
        if ($made.Code -ne 0 -or -not (Test-Path -LiteralPath $launcher)) {
            $made.Output | Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
            Stop-Install "could not write the launcher $launcher"
        }
        Write-Ok "launcher at $launcher"

        # The user's PATH, read as stored: %VARIABLES% in it stay unexpanded.
        $key = 'HKCU:\Environment'
        $userPath = (Get-Item -LiteralPath $key).GetValue('Path', '', 'DoNotExpandEnvironmentNames')
        $entries = @($userPath -split ';' | Where-Object { $_ })
        $onPath = $entries | Where-Object { [Environment]::ExpandEnvironmentVariables($_).TrimEnd('\') -ieq $binDir.TrimEnd('\') }
        $newTerminal = $false
        if ($onPath) {
            Write-Ok "$binDir is already on your PATH"
        } else {
            $value = (@($entries) + $binDir) -join ';'
            # As REG_EXPAND_SZ, what Windows makes it: written through .NET it would
            # become REG_SZ, and an entry such as %USERPROFILE%\bin would no longer
            # expand. Then a variable set and cleared through .NET, which is what
            # tells running programs (Explorer, new terminals) that it changed.
            $environment = [Microsoft.Win32.Registry]::CurrentUser.OpenSubKey('Environment', $true)
            try {
                $environment.SetValue('Path', $value, [Microsoft.Win32.RegistryValueKind]::ExpandString)
            } finally {
                $environment.Close()
            }
            [Environment]::SetEnvironmentVariable('DOMYN_INSTALLER_PATH_CHANGED', '1', 'User')
            [Environment]::SetEnvironmentVariable('DOMYN_INSTALLER_PATH_CHANGED', $null, 'User')
            Write-Ok "added $binDir to your PATH"
            $newTerminal = $true
        }
        # This window too, which `irm | iex` runs in.
        if (-not (($env:Path -split ';') | Where-Object { $_.TrimEnd('\') -ieq $binDir.TrimEnd('\') })) {
            $env:Path = "$binDir;$env:Path"
        }

        $runs = Invoke-Native $launcher @('--version')
        if ($runs.Code -ne 0) { Stop-Install "the installed launcher does not run: try $launcher --version" }
        Write-Ok "domyn $installed runs"

        # Commands run in Git Bash, never cmd.exe: without it domyn starts, and every
        # command says what to install.
        $bash = Invoke-Native $python @('-P', '-c', 'from domyn.agents.tools.builtin.git_bash import find_git_bash; print(find_git_bash())')
        if ($bash.Code -eq 0) {
            Write-Ok "Git Bash found: $($bash.Output | Select-Object -Last 1)"
        } else {
            Write-Warn 'Git for Windows is not installed: domyn runs commands in its Git Bash.'
            Write-Warn 'Install it from https://git-scm.com/download/win (or: winget install Git.Git).'
        }

        Write-Host ''
        Write-Host 'Ready.'
        if ($newTerminal) { Write-Host '  Other terminals see domyn once they are opened again.' }
        if ($planted -notcontains 'config.yaml') {
            Write-Host '  Then: ' -NoNewline; Write-Host 'domyn setup' -NoNewline; Write-Host '   (set up your agent)' -ForegroundColor DarkGray
            Write-Host '      ' -NoNewline
        }
        Write-Host '  domyn run' -NoNewline; Write-Host '     (start talking)' -ForegroundColor DarkGray
        Write-Host ''
        Write-Host "  runtime  $homeDir" -ForegroundColor DarkGray
        Write-Host "  config   $config" -ForegroundColor DarkGray
        Write-Host "  remove   Remove-Item -Recurse -Force '$homeDir', '$launcher'" -ForegroundColor DarkGray
        Write-Host ''
    } finally {
        Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
        if (-not $swapped -and (Test-Path -LiteralPath $incoming)) {
            Remove-Item -LiteralPath $incoming -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# Called last, so a download cut short by `irm | iex` runs nothing. Run as a file,
# a failure is the exit code (for scripts and CI); through iex, it is only said.
try {
    Install-Domyn @args
} catch {
    Write-Host 'fail ' -ForegroundColor Red -NoNewline
    Write-Host $_.Exception.Message
    if ($MyInvocation.MyCommand.Path) { exit 1 }
}

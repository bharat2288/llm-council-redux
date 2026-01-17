$WshShell = New-Object -ComObject WScript.Shell
$Desktop = [Environment]::GetFolderPath('Desktop')
$Shortcut = $WshShell.CreateShortcut("$Desktop\Council.lnk")
$Shortcut.TargetPath = "G:\My Drive\Work\Coded\LLM Council\Local Council\start_council.bat"
$Shortcut.WorkingDirectory = "G:\My Drive\Work\Coded\LLM Council\Local Council"
$Shortcut.IconLocation = "shell32.dll,21"
$Shortcut.Description = "Start Council Server"
$Shortcut.Save()
Write-Host "Desktop shortcut created: Council.lnk"

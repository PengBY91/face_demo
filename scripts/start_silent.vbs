Set WshShell = CreateObject("WScript.Shell")
' 获取脚本所在目录的父目录（项目根目录）
Set fso = CreateObject("Scripting.FileSystemObject")
scriptPath = fso.GetParentFolderName(WScript.ScriptFullName)
rootPath = fso.GetParentFolderName(scriptPath)
WshShell.CurrentDirectory = rootPath
WshShell.Run "manage.bat start", 0, False
Set WshShell = Nothing
Set fso = Nothing

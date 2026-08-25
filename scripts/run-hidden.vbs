Option Explicit

Dim sh, enginePath, scriptPath, profile, mode, command, exitCode
If WScript.Arguments.Count <> 4 Then
    WScript.Quit 2
End If

enginePath = WScript.Arguments(0)
scriptPath = WScript.Arguments(1)
profile = LCase(WScript.Arguments(2))
mode = LCase(WScript.Arguments(3))

If (profile <> "owner" And profile <> "partner") Then
    WScript.Quit 2
End If
If (mode <> "gateway" And mode <> "health") Then
    WScript.Quit 2
End If
If (InStr(enginePath, """) > 0 Or InStr(scriptPath, """) > 0) Then
    WScript.Quit 2
End If

command = """" & enginePath & """ -NoLogo -NoProfile -NonInteractive" & _
    " -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & scriptPath & _
    """ -Profile " & profile
If mode = "health" Then
    command = command & " -Repair"
End If

Set sh = CreateObject("WScript.Shell")
' Window style 0 prevents console flashes; waiting preserves the Scheduled
' Task exit status without exposing any stdout, message content, IDs or secrets.
exitCode = sh.Run(command, 0, True)
WScript.Quit exitCode

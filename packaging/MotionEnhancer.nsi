Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "WinVer.nsh"

!ifndef PAYLOAD
  !error "Pass /DPAYLOAD=<cmake install directory>"
!endif
!ifndef OUTPUT
  !error "Pass /DOUTPUT=<setup.exe>"
!endif
!ifndef VERSION
  !define VERSION "1.0.0"
!endif

Name "Motion Enhancer"
OutFile "${OUTPUT}"
InstallDir "$LOCALAPPDATA\Programs\MotionEnhancer"
InstallDirRegKey HKCU "Software\MotionEnhancer" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 32
ShowInstDetails show
ShowUninstDetails show
BrandingText "Motion Enhancer ${VERSION} (x64)"
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "Motion Enhancer"
VIAddVersionKey "FileDescription" "Motion Enhancer installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Motion Enhancer contributors"

!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\motion_enhancer.exe"
!define MUI_FINISHPAGE_RUN_PARAMETERS "--gui"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "${PAYLOAD}\LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
  SetShellVarContext current
  SetRegView 64
  ${IfNot} ${RunningX64}
  ${OrIfNot} ${AtLeastWin10}
    MessageBox MB_ICONSTOP "Motion Enhancer requires Windows 10/11 x64."
    Abort
  ${EndIf}
FunctionEnd

Section "Application, AI models, runtimes and trainer files (required)" App
  SectionIn RO
  SetOutPath "$INSTDIR"
  IfFileExists "$INSTDIR\motion_enhancer.exe" 0 unlocked
  ClearErrors
  FileOpen $0 "$INSTDIR\motion_enhancer.exe" a
  IfErrors locked
  FileClose $0
  Goto unlocked
locked:
  MessageBox MB_ICONSTOP "Close Motion Enhancer before installing this update." /SD IDOK
  SetErrorLevel 1
  Abort
unlocked:
  SetOverwrite on
  File /r /x models "${PAYLOAD}\*"
  ; Preserve models trained/exported by the user during repair or upgrade.
  SetOverwrite off
  SetOutPath "$INSTDIR\models"
  File /r "${PAYLOAD}\models\*"
  SetOverwrite on
  SetOutPath "$INSTDIR"
  CreateDirectory "$SMPROGRAMS\Motion Enhancer"
  CreateShortcut "$SMPROGRAMS\Motion Enhancer\Motion Enhancer.lnk" "$INSTDIR\motion_enhancer.exe" "--gui"
  CreateShortcut "$SMPROGRAMS\Motion Enhancer\Set up AI Trainer.lnk" "$INSTDIR\Setup-AITrainer.cmd"
  CreateShortcut "$SMPROGRAMS\Motion Enhancer\Installation guide.lnk" "$INSTDIR\INSTALLATION.md"
  CreateShortcut "$SMPROGRAMS\Motion Enhancer\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\MotionEnhancer" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "DisplayName" "Motion Enhancer"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "Publisher" "Motion Enhancer contributors"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "DisplayIcon" "$INSTDIR\motion_enhancer.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "UninstallString" '$\"$INSTDIR\Uninstall.exe$\"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "QuietUninstallString" '$\"$INSTDIR\Uninstall.exe$\" /S'
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "NoModify" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "NoRepair" 1
  WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer" "EstimatedSize" ${SIZE_KB}
SectionEnd

Section /o "Set up AI trainer dependencies (Internet, several GB)" Trainer
  DetailPrint "Installing trainer dependencies; see the separate setup window."
  ExecWait '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\tools\ai-train\Setup-AITrainer.ps1"' $0
  ${If} $0 != 0
    DetailPrint "Trainer setup failed (code $0). The application is installed; retry Set up AI Trainer later."
    MessageBox MB_ICONEXCLAMATION "Trainer setup did not finish. The application is installed and can use its bundled models. Run Set up AI Trainer to retry." /SD IDOK
  ${EndIf}
SectionEnd

Function un.onInit
  SetShellVarContext current
  SetRegView 64
  IfFileExists "$INSTDIR\motion_enhancer.exe" 0 unlocked
  ClearErrors
  FileOpen $0 "$INSTDIR\motion_enhancer.exe" a
  IfErrors locked
  FileClose $0
  Goto unlocked
locked:
  MessageBox MB_ICONSTOP "Close Motion Enhancer before uninstalling it." /SD IDOK
  SetErrorLevel 1
  Abort
unlocked:
FunctionEnd

Section "Uninstall"
  ; Generated exact file list: never recursively delete user data or models.
  !include "${UNINSTALL_LIST}"
  Delete "$INSTDIR\Uninstall.exe"
  Delete "$SMPROGRAMS\Motion Enhancer\Motion Enhancer.lnk"
  Delete "$SMPROGRAMS\Motion Enhancer\Set up AI Trainer.lnk"
  Delete "$SMPROGRAMS\Motion Enhancer\Installation guide.lnk"
  Delete "$SMPROGRAMS\Motion Enhancer\Uninstall.lnk"
  RMDir "$SMPROGRAMS\Motion Enhancer"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\MotionEnhancer"
  DeleteRegKey HKCU "Software\MotionEnhancer"
  RMDir "$INSTDIR"
SectionEnd

Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "WinVer.nsh"
!include "FileFunc.nsh"

!ifndef THEME_PLUGIN
  !error "Pass /DTHEME_PLUGIN=<directory containing the Win32 MotionTheme.dll>"
!endif
!addplugindir /x86-unicode "${THEME_PLUGIN}"

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
!define MUI_ICON "assets\installer.ico"
!define MUI_UNICON "assets\installer.ico"
!define MUI_BGCOLOR "08080A"
!define MUI_TEXTCOLOR "EEEEF2"
!define MUI_LICENSEPAGE_BGCOLOR "0D0D0F"
!define MUI_INSTFILESPAGE_COLORS "EEEEF2 0D0D0F"
!define MUI_COMPONENTSPAGE_NODESC
!define MUI_CUSTOMFUNCTION_GUIINIT ThemeInit
!define MUI_CUSTOMFUNCTION_UNGUIINIT un.ThemeInit
!define MUI_FINISHPAGE_RUN "$INSTDIR\motion_enhancer.exe"
!define MUI_FINISHPAGE_RUN_PARAMETERS "--gui"
!define MUI_FINISHPAGE_RUN_NOTCHECKED
!define MUI_WELCOMEPAGE_TITLE "A smoother experience."
!define MUI_WELCOMEPAGE_TEXT "Real-time frame generation, built for your GPU.$\r$\n$\r$\nINCLUDED IN THIS INSTALL$\r$\n$\r$\n    Motion Enhancer and compiled GPU shaders$\r$\n    AMD FidelityFX Optical Flow$\r$\n    AI flow and depth models$\r$\n    ONNX Runtime, DirectML and Microsoft runtimes$\r$\n$\r$\nThe AI trainer is optional. Existing models work offline.$\r$\n$\r$\nUse Music or Alt+M to toggle the ambient soundtrack."
!define MUI_PAGE_CUSTOMFUNCTION_SHOW ThemeWelcome
!insertmacro MUI_PAGE_WELCOME
!define MUI_PAGE_CUSTOMFUNCTION_SHOW ThemeLicense
!insertmacro MUI_PAGE_LICENSE "${PAYLOAD}\LICENSE"
!define MUI_COMPONENTSPAGE_TEXT_TOP "All application runtime dependencies are included. Trainer setup is optional and downloads several GB of GPU-specific packages."
!define MUI_PAGE_CUSTOMFUNCTION_SHOW ThemeComponents
!insertmacro MUI_PAGE_COMPONENTS
!define MUI_PAGE_CUSTOMFUNCTION_SHOW ThemeDirectory
!insertmacro MUI_PAGE_DIRECTORY
!define MUI_PAGE_CUSTOMFUNCTION_SHOW ThemeInstall
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TITLE "You're ready to flow."
!define MUI_FINISHPAGE_TEXT "The app, GPU runtimes and existing AI models are installed.$\r$\n$\r$\nOpen Motion Enhancer, select a window, and start generating frames.$\r$\n$\r$\nYou can set up the optional AI trainer later from the Start menu."
!define MUI_FINISHPAGE_RUN_TEXT "Launch Motion Enhancer"
!define MUI_PAGE_CUSTOMFUNCTION_SHOW ThemeFinish
!insertmacro MUI_PAGE_FINISH
!define MUI_PAGE_CUSTOMFUNCTION_SHOW un.ThemeConfirm
!insertmacro MUI_UNPAGE_CONFIRM
!define MUI_INSTFILESPAGE_FINISHHEADER_TEXT "Application removed."
!define MUI_INSTFILESPAGE_FINISHHEADER_SUBTEXT "Your models, captures and training data have been preserved."
!define MUI_PAGE_CUSTOMFUNCTION_SHOW un.ThemeRemove
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Var MusicEnabled

Function ThemeInit
  ; GUI-only extraction/playback: /S never loads the skin or starts audio.
  InitPluginsDir
  File "/oname=$PLUGINSDIR\setup-music.wav" "assets\setup-music.wav"
  MotionTheme::Attach /NOUNLOAD "$PLUGINSDIR\setup-music.wav" "$MusicEnabled" "0"
FunctionEnd

Function un.ThemeInit
  ; Match the app on removal as well, without automatically playing music.
  MotionTheme::Attach /NOUNLOAD "" "0" "1"
FunctionEnd

Function ThemeWelcome
  MotionTheme::Page /NOUNLOAD "$mui.WelcomePage" "0" "A smoother experience." "Motion Enhancer / Windows x64"
FunctionEnd
Function ThemeLicense
  MotionTheme::Page /NOUNLOAD "$mui.LicensePage" "1" "Built on open technology." "Review the license before continuing."
FunctionEnd
Function ThemeComponents
  MotionTheme::Page /NOUNLOAD "$mui.ComponentsPage" "2" "Everything you need." "Choose your installation components."
FunctionEnd
Function ThemeDirectory
  MotionTheme::Page /NOUNLOAD "$mui.DirectoryPage" "3" "Find its home." "Install for this Windows account. No administrator required."
FunctionEnd
Function ThemeInstall
  MotionTheme::Page /NOUNLOAD "$mui.InstFilesPage" "4" "Bringing motion to life." "Installing the app, AI models and GPU runtimes."
FunctionEnd
Function ThemeFinish
  MotionTheme::Page /NOUNLOAD "$mui.FinishPage" "5" "You're ready to flow." "Your setup is complete."
FunctionEnd
Function un.ThemeConfirm
  MotionTheme::Page /NOUNLOAD "$mui.UnConfirmPage" "0" "Make room for what's next." "Models, captures and training data will be preserved."
FunctionEnd
Function un.ThemeRemove
  MotionTheme::Page /NOUNLOAD "$mui.InstFilesPage" "1" "Removing Motion Enhancer." "Your models, captures and training data stay with you."
FunctionEnd

Function .onGUIEnd
  MotionTheme::Stop /NOUNLOAD
FunctionEnd
Function un.onGUIEnd
  MotionTheme::Stop /NOUNLOAD
FunctionEnd

Function un.onUninstSuccess
  IfSilent done
  MotionTheme::Page /NOUNLOAD "$mui.InstFilesPage" "2" "Application removed." "Your models, captures and training data have been preserved."
done:
FunctionEnd

Function .onInit
  SetShellVarContext current
  SetRegView 64
  StrCpy $MusicEnabled "1"
  ${GetParameters} $0
  ClearErrors
  ${GetOptions} $0 "/NOMUSIC" $1
  ${IfNot} ${Errors}
    StrCpy $MusicEnabled "0"
  ${EndIf}
  ${IfNot} ${RunningX64}
  ${OrIfNot} ${AtLeastWin10}
    MessageBox MB_ICONSTOP "Motion Enhancer requires Windows 10/11 x64."
    Abort
  ${EndIf}
FunctionEnd

Section "Motion Enhancer, AI models and GPU runtimes (required)" App
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

import subprocess, sys

Windows_Only = False
Build = 'Test'

def print亮绿(*kw,**kargs):
    print("\033[1;32m",*kw,"\033[0m",**kargs)

if not Windows_Only:
    process = subprocess.Popen([
        "C:/Program Files/FreeFileSync/FreeFileSync.exe",
        "Win2Linux.ffs_batch"
    ])
    process.wait()
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* Finish Win2Linux ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')

    process = subprocess.Popen([
        "C:/Program Files/FreeFileSync/FreeFileSync.exe",
        "F:/UHMP_LINUX/CopyPackageToServer.ffs_batch"
    ])
    process.wait()
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* CopyPackageToServer ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')





print亮绿('********* Begin build windows renderer ***********')
# build Visual
process = subprocess.Popen([
    "{EnginePath}/Build/BatchFiles/RunUAT.bat",
    "-ScriptsForProject=F:/UHMP/UHMP.uproject",
    "BuildCookRun",
    "-nocompileeditor",
    "-nop4",
    "-project=F:/UHMP/UHMP.uproject",
    "-cook",
    "-stage",
    "-archive",
    "-archivedirectory=F:/UHMP/Build",
    "-package",
    "-ue4exe={EnginePath}/Binaries/Win64/UE4Editor-Cmd.exe",
    "-compressed",
    "-ddc=DerivedDataBackendGraph",
    "-pak",
    "-prereqs",
    "-nodebuginfo",
    "-targetplatform=Win64",
    "-build",
    "-target=UHMP",
    "-clientconfig=%s"%Build,
    "-utf8output",
    "-compile",
])
return_code = process.wait()
print亮绿('********* ********************** ***********')
print亮绿('********* ********************** ***********')
print亮绿('********* End build windows renderer ***********')
print亮绿('********* ********************** ***********')
print亮绿('********* ********************** ***********')
if (return_code!=0):
    print('fail')
    sys.exit()





print亮绿('********* Begin build windows server ***********')
# build server
process = subprocess.Popen([
    "{EnginePath}/Build/BatchFiles/RunUAT.bat",
    "-ScriptsForProject=F:/UHMP/UHMP.uproject",  
    "BuildCookRun",
    "-nocompileeditor",
    "-nop4",
    "-project=F:/UHMP/UHMP.uproject",
    "-cook",
    "-stage",
    "-archive",
    "-archivedirectory=F:/UHMP/Build",
    "-package ",
    "-ue4exe={EnginePath}/Binaries/Win64/UE4Editor-Cmd.exe",
    "-compressed",
    "-ddc=DerivedDataBackendGraph ",
    "-pak",
    "-prereqs",
    "-nodebuginfo",
    "-targetplatform=Win64",
    "-build",
    "-target=UHMPServer",
    "-serverconfig=%s"%Build,
    "-utf8output",
    "-compile"
])
return_code = process.wait()
print亮绿('********* ********************** ***********')
print亮绿('********* ********************** ***********')
print亮绿('********* End build windows server ***********')
print亮绿('********* ********************** ***********')
print亮绿('********* ********************** ***********')
if (return_code!=0):
    print('fail')
    sys.exit()




if not Windows_Only:

    print亮绿('********* Begin build linux server ***********')
    # build linux server
    process = subprocess.Popen([
        "{EnginePath}/Build/BatchFiles/RunUAT.bat",
        "-ScriptsForProject=F:/UHMP_LINUX/UHMP.uproject",
        "BuildCookRun",
        "-nocompileeditor",
        "-nop4",
        "-project=F:/UHMP_LINUX/UHMP.uproject",
        "-cook",
        "-stage",
        "-archive",
        "-archivedirectory=F:/UHMP_LINUX/Build",
        "-package",
        "-ue4exe={EnginePath}/Binaries/Win64/UE4Editor-Cmd.exe",
        "-compressed",
        "-ddc=DerivedDataBackendGraph",
        "-pak",
        "-prereqs",
        "-nodebuginfo",
        "-targetplatform=Linux",
        "-build",
        "-target=UHMPServer",
        "-serverconfig=%s"%Build,
        "-utf8output",
        "-compile"
    ])
    return_code = process.wait()
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* End build linux server ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    if (return_code!=0):
        print('fail')
        sys.exit()




    print亮绿('********* Begin build linux renderer ***********')
    # build linux renderer
    process = subprocess.Popen([
        "{EnginePath}/Build/BatchFiles/RunUAT.bat",
        "-ScriptsForProject=F:/UHMP_LINUX/UHMP.uproject",
        "BuildCookRun",
        "-nocompileeditor",
        "-nop4",
        "-project=F:/UHMP_LINUX/UHMP.uproject",
        "-cook",
        "-stage",
        "-archive",
        "-archivedirectory=F:/UHMP_LINUX/Build",
        "-package",
        "-ue4exe={EnginePath}/Binaries/Win64/UE4Editor-Cmd.exe",
        "-compressed",
        "-ddc=DerivedDataBackendGraph",
        "-pak",
        "-prereqs",
        "-nodebuginfo",
        "-targetplatform=Linux",
        "-build",
        "-target=UHMP",
        "-clientconfig=%s"%Build,
        "-utf8output",
        "-compile",
    ])
    return_code = process.wait()
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* End build linux renderer  ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    if (return_code!=0):
        print('fail')
        sys.exit()






    process = subprocess.Popen([
        "C:/Program Files/FreeFileSync/FreeFileSync.exe",
        "Win2Linux.ffs_batch"
    ])
    process.wait()
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* Win2Linux ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')

    process = subprocess.Popen([
        "C:/Program Files/FreeFileSync/FreeFileSync.exe",
        "F:/UHMP_LINUX/CopyPackageToServer.ffs_batch"
    ])
    process.wait()
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* CopyPackageToServer ***********')
    print亮绿('********* ********************** ***********')
    print亮绿('********* ********************** ***********')




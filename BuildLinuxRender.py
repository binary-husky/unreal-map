import subprocess, sys, shutil, time, os

EnginePath = "F:/UnrealSourceCode/UnrealEngine-4.27.2-release/Engine"
assert os.path.exists(EnginePath), f"Cannot find Unreal Engine at this path {EnginePath}"
Windows_Only = False
Build = 'Test' # Development/Test/shipping
Platform = 'Linux'  # Win64/Linux
def print亮绿(*kw,**kargs):
    print("\033[1;32m",*kw,"\033[0m",**kargs)

time_mark = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
try:
    shutil.rmtree('Build/LinuxNoEditor')
except:
    pass

print亮绿(f'********* Begin Build: {Build} On {Platform} ***********')
# build server
path = os.path.abspath('./').replace(r'\\', '/')

process = subprocess.Popen([
    f"{EnginePath}/Build/BatchFiles/RunUAT.bat",
    f"-ScriptsForProject={path}/UHMP.uproject",  
    "BuildCookRun",
    "-nocompileeditor",
    "-nop4",
    f"-project={path}/UHMP.uproject",
    "-cook",
    "-stage",
    "-archive",
    f"-archivedirectory={path}/Build",
    "-package ",
    f"-ue4exe={EnginePath}/Binaries/Win64/UE4Editor-Cmd.exe",
    "-compressed",
    "-ddc=DerivedDataBackendGraph",
    "-pak",
    "-prereqs",
    "-nodebuginfo",
    f"-targetplatform={Platform}",
    "-build",
    "-target=UHMP",
    "-serverconfig=%s"%Build,
    "-utf8output",
    "-compile"
])
return_code = process.wait()
print亮绿('********* ********************** ***********')
print亮绿('********* ********************** ***********')
print亮绿('********* ********************** ***********')
print亮绿('********* ********************** ***********')
if (return_code!=0):
    print('fail')
    sys.exit()


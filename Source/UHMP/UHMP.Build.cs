// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.IO;

public class UHMP : ModuleRules
{
	public UHMP(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
	
		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "Sockets", "Networking", "Jsonx", "JsonxUtilities", "XtensorAPI" });

		PrivateDependencyModuleNames.AddRange(new string[] { });

        CMakeTarget.add(Target, this, "ipc", Path.Combine(this.ModuleDirectory, "../Deps/cppipc"), "");
		
		bEnableExceptions = true;


	}
}

// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class JsonxUtilities : ModuleRules
{
	public JsonxUtilities( ReadOnlyTargetRules Target ) : base(Target)
	{
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Jsonx",
			}
		);
	}
}

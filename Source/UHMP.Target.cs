// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.Collections.Generic;

public class UHMPTarget : TargetRules
{
	public UHMPTarget(TargetInfo Target) : base(Target)
	{
		Type = TargetType.Game;
		DefaultBuildSettings = BuildSettingsVersion.V2;
		bUseChecksInShipping = true;

		ExtraModuleNames.AddRange( new string[] { "UHMP" } );
	}
}

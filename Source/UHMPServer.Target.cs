// Fill out your copyright notice in the Description page of Project Settings.

using UnrealBuildTool;
using System.Collections.Generic;

public class UHMPServerTarget : TargetRules //Change this line according to the name of your project
{
    public UHMPServerTarget(TargetInfo Target) : base(Target) //Change this line according to the name of your project
    {
        Type = TargetType.Server;
        DefaultBuildSettings = BuildSettingsVersion.V2;
        bUseChecksInShipping = true;
        ExtraModuleNames.Add("UHMP"); //Change this line according to the name of your project
    }
}
// Fill out your copyright notice in the Description page of Project Settings.

#pragma once


#include "CoreMinimal.h"
#include "Networking.h"
#include "Engine/LevelScriptActor.h"
#include "Containers/UnrealString.h"
#include "HMPLevelScriptActor.generated.h"

/**
 *
 */

USTRUCT(BlueprintType)
struct FAgentSetting
{
	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString ClassName = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int AgentTeam = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int IndexInTeam = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int UID = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool AcceptRLControl = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float MaxMoveSpeed = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector InitLocation;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector InitRotation;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector AgentScale;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector InitVelocity;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float AgentHp;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float WeaponCD = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float MaxEpisodeStep = 999;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool IsTeamReward = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString Type = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString InitBuff = "";
};



USTRUCT(BlueprintType)
struct FParsedTcpInData
{
	// please change lines in 
	// bool AHMPLevelScriptActor::ParsedTcpInData()
	// together with this struct

	GENERATED_USTRUCT_BODY()


	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool valid = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString DataCmd;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int NumAgents = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FAgentSetting> AgentSettingArray;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int TimeStep = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<float> Actions;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FString> StringActions;

};


USTRUCT(BlueprintType)
struct FTcpOutAgentData
{
	// please change lines in 
	// bool AHMPLevelScriptActor::ParsedTcpInData()
	// together with this struct

	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool Valid = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool AgentAlive = true;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int AgentTeam = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int IndexInTeam = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int UID = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool AcceptRLControl = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float MaxMoveSpeed = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector AgentLocation;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector AgentRotation;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector AgentScale;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FVector AgentVelocity;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float AgentHp;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float WeaponCD = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float MaxEpisodeStep = 999;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int TimeCnt = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float Time = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int PreviousAction;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<int> AvailActions;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float Reward;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool IsTeamReward = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool EpisodeDone = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString Type = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString CurrentBuff = "";
};


USTRUCT(BlueprintType)
struct FTcpOutAgentDataArr
{
	// please change lines in 
	// bool AHMPLevelScriptActor::ParsedTcpInData()
	// together with this struct

	GENERATED_USTRUCT_BODY()

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool Valid = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FTcpOutAgentData> DataArr;
};

UCLASS()
class UHMP_API AHMPLevelScriptActor : public ALevelScriptActor
{
	GENERATED_BODY()

};



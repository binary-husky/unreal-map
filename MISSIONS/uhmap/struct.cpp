#pragma once

#include "CoreMinimal.h"
#include "Containers/UnrealString.h"
#include "DataStruct.generated.h"

USTRUCT(BlueprintType)
struct FAgentProperty
{
	GENERATED_BODY()

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
		bool IsTeamReward = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString Type = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString WeaponType = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString Color = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float DodgeProb = 0.0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float ExplodeDmg = 20.0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float FireRange = 1000.0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float GuardRange = 1400.0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD1 = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD2 = "";
};



USTRUCT(BlueprintType)
struct FParsedDataInput
{
	// please change lines in 
	// bool AHMPLevelScriptActor::ParsedTcpInData()
	// together with this struct

	GENERATED_BODY()


	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool valid = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString DataCmd;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int NumAgents = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FAgentProperty> AgentSettingArray;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int TimeStep = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int TimeStepMax = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<float> Actions;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FString> StringActions;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD1 = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD2 = "";
};



USTRUCT(BlueprintType)
struct FAgentDataOutput
{
	// please change lines in 
	// bool AHMPLevelScriptActor::ParsedTcpInData()
	// together with this struct

	GENERATED_BODY()

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
		int PreviousAction;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<int> AvailActions;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float Reward;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool IsTeamReward = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FString> Interaction;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString Type = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD1 = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD2 = "";

};

USTRUCT(BlueprintType)
struct FGlobalDataOutput
{

	GENERATED_BODY()

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool Valid = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float TeamReward = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool UseTeamReward = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FString> Events;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<int> VisibleMatFlatten;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<float> DisMatFlatten;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float MaxEpisodeStep = 999;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		int TimeCnt = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		float Time = 0;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool EpisodeDone = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD1 = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVD2 = "";

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FString RSVDJson = "";
};

USTRUCT(BlueprintType)
struct FAgentDataOutputArr
{
	// please change lines in 
	// bool AHMPLevelScriptActor::ParsedTcpInData()
	// together with this struct

	GENERATED_BODY()

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		bool Valid = false;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		TArray<FAgentDataOutput> DataArr;

	UPROPERTY(EditDefaultsOnly, BlueprintReadWrite)
		FGlobalDataOutput DataGlobal;
};

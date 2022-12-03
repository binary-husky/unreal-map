// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Stack.h"
#include "Misc/UObjectToken.h"
#include "Kismet/KismetSystemLibrary.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "Kismet/GameplayStatics.h"
#include "Navigation/CrowdFollowingComponent.h"
#include "Runtime/CoreUObject/Public/UObject/NoExportTypes.h"
#include "GenericTeamAgentInterface.h"
#include "AIController.h"
#include "AgentBaseCpp.h"
//
//#pragma push_macro("NDEBUG")
//#undef NDEBUG
//#include <cassert>
//#define assertm(exp, msg) assert(((void)msg, exp))
//#pragma pop_macro("NDEBUG")
//

#include "UHMPBlueprintFunctionLibrary.generated.h"

/**
 * 
 */
UCLASS()
class UHMP_API UUHMPBlueprintFunctionLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()
public:
	UFUNCTION(BlueprintCallable, Category = "SpecialFn")
		static void PrintStringSpecial(FString string);

	UFUNCTION(BlueprintCallable, Category = "SpecialFn")
		static void AssertFalse(FString string);


	UFUNCTION(BlueprintPure , Category = "SpecialFn")
		static FVector VectorClearZ(FVector Vector);

	UFUNCTION(BlueprintPure, Category = "SpecialFn")
		static bool IsAllZeroVector(FVector Vector);

	UFUNCTION(BlueprintPure, Category = "SpecialFn")
		static bool IsZero(int x);

	UFUNCTION(BlueprintPure, Category = "SpecialFn")
		static bool IsZeroFloat(float x);

	UFUNCTION(BlueprintCallable, meta = (WorldContext = "WorldContextObject", CallableWithoutWorldContext, Keywords = "raise error", DevelopmentOnly), Category = "Utilities|Debugging")
		static void RaiseError(UObject* WorldContextObject, const FString& ErrorMessage = FString(TEXT("An error occurred")), bool bPrintToOutputLog = true);
	
	UFUNCTION(BlueprintCallable, meta = (WorldContext = "WorldContextObject", CallableWithoutWorldContext, Keywords = "raise fatal error"), Category = "Utilities|Debugging")
		static void RaiseFatalError(UObject* WorldContextObject, const FString& ErrorMessage = FString(TEXT("Hmap Fatal error occurred ~_~")));

	UFUNCTION(BlueprintPure, Category = "SpecialFn")
		static float GetFrameRatePerGameSecond();

	UFUNCTION(BlueprintCallable, Category = "SpecialFn", meta = (WorldContext = "WorldContextObject", CallableWithoutWorldContext))
		static float GetSimDeltaTime(const UObject* WorldContextObject);

	UFUNCTION(BlueprintCallable, Category = "AI")
		static void SetAITeamForPerceptionFilter(AAIController* Controller, const FGenericTeamId& NewTeamID);

	UFUNCTION(BlueprintCallable, Category = "AI")
		static FGenericTeamId GetAITeamForPerceptionFilter(AAIController* Controller);

	UFUNCTION(BlueprintCallable, Category = "AI")
		static TArray<int> GetAffilationArray(const TArray<AAgentBaseCpp*> &agents);

	UFUNCTION(BlueprintCallable, Category = "AI")
		static TArray<float> GetPerceptionRangeArray(const TArray<AAgentBaseCpp*>& agents);

	UFUNCTION(BlueprintCallable, Category = "SpecialFn")
		static void MannualGc();

	UFUNCTION(BlueprintPure, Category = "SpecialFn")
		static FVector FlyingTracking(FVector self_pos, FVector dst_pos, bool maintain_z, float dis_aim);
};


//UFUNCTION(BlueprintCallable, Category = "ActorFuncions", meta = (DeterminesOutputType = "InputActor"))
//	  static AActor* CloneActor(AActor* InputActor);
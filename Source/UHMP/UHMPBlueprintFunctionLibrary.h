// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/Stack.h"
#include "Misc/UObjectToken.h"
#include "EngineUtils.h"
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



	UFUNCTION(BlueprintCallable, Category = "SpecialFn")
		static void MannualGc();

	UFUNCTION(BlueprintCallable, Category = "SpecialFn")
		static bool IsEditor();

	UFUNCTION(BlueprintPure, Category = "SpecialFn")
		static FVector FlyingTracking(FVector self_pos, FVector dst_pos, bool maintain_z, float dis_aim);

	UFUNCTION(BlueprintCallable, Category = "SpecialFn")
		static void SortActorListBy(TArray<AActor*> InActors, TArray<float> ScoreList, TArray<AActor*>& OutActors);

	//UFUNCTION(BlueprintCallable, Category = "Utilities", meta = (WorldContext = "WorldContextObject", DeterminesOutputType = "ActorClass", DynamicOutputParam = "OutActors"))
	//	static void MyGetAllActorsOfClass(const UObject* WorldContextObject, TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors);

	UFUNCTION(BlueprintCallable, Category = "SpecialFn", meta = (WorldContext = "WorldContextObject", DeterminesOutputType = "ActorClass", DynamicOutputParam = "OutActors"))
		static void TarrayChangeClass(TArray<AActor*> InActors, TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors);

	/**
	 *	Find all Actors in the world of the specified class.
	 *	This is a slow operation, use with caution e.g. do not use every frame.
	 *	@param	ActorClass	Class of Actor to find. Must be specified or result array will be empty.
	 *	@param	OutActors	Output array of Actors of the specified class.
	 */
	UFUNCTION(BlueprintCallable, Category = "Utilities", meta = (WorldContext = "WorldContextObject", DeterminesOutputType = "ActorClass", DynamicOutputParam = "OutActors"))
		static void GetAllActorsOfClassWithOrder(const UObject* WorldContextObject, TSubclassOf<AActor> ActorClass, TArray<AActor*>& OutActors);
	/**
	 *	Find all Actors in the world with the specified tag.
	 *	This is a slow operation, use with caution e.g. do not use every frame.
	 *	@param	Tag			Tag to find. Must be specified or result array will be empty.
	 *	@param	OutActors	Output array of Actors of the specified tag.
	 */
	UFUNCTION(BlueprintCallable, Category = "Utilities", meta = (WorldContext = "WorldContextObject"))
		static void GetAllActorsWithTagWithOrder(const UObject* WorldContextObject, FName Tag, TArray<AActor*>& OutActors);

	/**
	 *	Find all Actors in the world of the specified class with the specified tag.
	 *	This is a slow operation, use with caution e.g. do not use every frame.
	 *	@param	Tag			Tag to find. Must be specified or result array will be empty.
	 *	@param	ActorClass	Class of Actor to find. Must be specified or result array will be empty.
	 *	@param	OutActors	Output array of Actors of the specified tag.
	 */
	UFUNCTION(BlueprintCallable, Category = "Utilities", meta = (WorldContext = "WorldContextObject", DeterminesOutputType = "ActorClass", DynamicOutputParam = "OutActors"))
		static void GetAllActorsOfClassWithTagWithOrder(const UObject* WorldContextObject, TSubclassOf<AActor> ActorClass, FName Tag, TArray<AActor*>& OutActors);

};


//UFUNCTION(BlueprintCallable, Category = "ActorFuncions", meta = (DeterminesOutputType = "InputActor"))
//	  static AActor* CloneActor(AActor* InputActor);
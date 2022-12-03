// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "UObject/WeakInterfacePtr.h"
#include "EngineDefines.h"
#include "AITypes.h"
#include "Navigation/PathFollowingComponent.h"
#include "Navigation/CrowdAgentInterface.h"
#include "AI/Navigation/NavigationAvoidanceTypes.h"
#include "Navigation/CrowdFollowingComponent.h"
#include "HmpCrowdFollowingComponent.generated.h"

/**
 * 
 */
UCLASS(ClassGroup = AI, HideCategories = (Activation, Collision), meta = (BlueprintSpawnableComponent), config = Game)
class UHMP_API UHmpCrowdFollowingComponent : public UCrowdFollowingComponent
{
	GENERATED_BODY()

protected:
	void UpdatePathSegment() override;
	
};

// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "GenericTeamAgentInterface.h"
#include "AgentBaseCpp.generated.h"

UCLASS()
class UHMP_API AAgentBaseCpp : public ACharacter, public IGenericTeamAgentInterface
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	AAgentBaseCpp();
	UPROPERTY(BlueprintReadWrite)
		FGenericTeamId GenericTeamNo;

	UPROPERTY(BlueprintReadWrite)
		float PerceptionRange = 1500;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;



public:	

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;


	virtual FGenericTeamId GetGenericTeamId() const override { return GenericTeamNo; }


};

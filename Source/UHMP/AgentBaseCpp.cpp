// Fill out your copyright notice in the Description page of Project Settings.


#include "AgentBaseCpp.h"

// Sets default values
AAgentBaseCpp::AAgentBaseCpp()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AAgentBaseCpp::BeginPlay()
{
	Super::BeginPlay();
	
}


// Called to bind functionality to input
void AAgentBaseCpp::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

}


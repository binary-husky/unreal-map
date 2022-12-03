// Fill out your copyright notice in the Description page of Project Settings.
#include "HmpCrowdFollowingComponent.h"
#include "Navigation/MetaNavMeshPath.h"

void UHmpCrowdFollowingComponent::UpdatePathSegment()
{
	if (GEngine)
	{
		GEngine->AddOnScreenDebugMessage(-1, 5.0f, FColor::Blue, "Hack is Done");
	}
	if (!IsCrowdSimulationEnabled())
	{
		Super::UpdatePathSegment();
		return;
	}

	if (!Path.IsValid() || MovementComp == NULL)
	{
		OnPathFinished(FPathFollowingResult(EPathFollowingResult::Aborted, FPathFollowingResultFlags::InvalidPath));
		return;
	}

	if (!Path->IsValid())
	{
		if (!Path->IsWaitingForRepath())
		{
			//UE_VLOG(this, LogPathFollowing, Log, TEXT("Aborting move due to path being invalid and not waiting for repath"));
			OnPathFinished(FPathFollowingResult(EPathFollowingResult::Aborted, FPathFollowingResultFlags::InvalidPath));
			return;
		}
		else
		{
			// continue with execution, if navigation is being rebuild constantly AI will get stuck with current waypoint
			// path updates should be still coming in, even though they get invalidated right away
			//UE_VLOG(this, LogPathFollowing, Log, TEXT("Updating path points in invalid & pending path!"));
		}
	}

	// if agent has control over its movement, check finish conditions
	const FVector CurrentLocation = MovementComp->GetActorFeetLocation();
	const bool bCanReachTarget = MovementComp->CanStopPathFollowing();
	if (bCanReachTarget && Status == EPathFollowingStatus::Moving)
	{
		const FVector GoalLocation = GetCurrentTargetLocation();

		if (bCollidedWithGoal)
		{
			// check if collided with goal actor
			OnSegmentFinished();
			OnPathFinished(FPathFollowingResult(EPathFollowingResult::Success, FPathFollowingResultFlags::None));
		}
		else if (bFinalPathPart)
		{
			const FVector ToTarget = (GoalLocation - MovementComp->GetActorFeetLocation());
			const bool bMovedTooFar = false; // (bCheckMovementAngle || bCanCheckMovingTooFar) && !CrowdAgentMoveDirection.IsNearlyZero() && FVector::DotProduct(ToTarget, CrowdAgentMoveDirection) < 0.0;

//#if ENABLE_VISUAL_LOG
//			if (bMovedTooFar)
//			{
//				const FVector AgentLoc = MovementComp->GetActorFeetLocation();
//				UE_VLOG_SEGMENT(GetOwner(), LogCrowdFollowing, Log, AgentLoc, AgentLoc + CrowdAgentMoveDirection * 100.0f, FColor::Cyan, TEXT("moveDir"));
//				UE_VLOG_SEGMENT(GetOwner(), LogCrowdFollowing, Log, AgentLoc, AgentLoc + ToTarget.GetSafeNormal() * 100.0f, FColor::Cyan, TEXT("toTarget"));
//				UE_VLOG(GetOwner(), LogCrowdFollowing, Log, TEXT("Moved too far, dotValue: %.2f (normalized dot: %.2f) velocity:%s (speed:%.0f)"),
//					FVector::DotProduct(ToTarget, CrowdAgentMoveDirection),
//					FVector::DotProduct(ToTarget.GetSafeNormal(), CrowdAgentMoveDirection),
//					*MovementComp->Velocity.ToString(),
//					MovementComp->Velocity.Size()
//				);
//			}
//#endif

			// can't use HasReachedDestination here, because it will use last path point
			// which is not set correctly for partial paths without string pulling
			const float UseAcceptanceRadius = GetFinalAcceptanceRadius(*Path, OriginalMoveRequestGoalLocation, &GoalLocation);
			if (bMovedTooFar || HasReachedInternal(GoalLocation, 0.0f, 0.0f, CurrentLocation, UseAcceptanceRadius, bReachTestIncludesAgentRadius ? MinAgentRadiusPct : 0.0f))
			{
				//UE_VLOG(GetOwner(), LogCrowdFollowing, Log, TEXT("Last path segment finished due to \'%s\'"), bMovedTooFar ? TEXT("Missing Last Point") : TEXT("Reaching Destination"));
				OnPathFinished(FPathFollowingResult(EPathFollowingResult::Success, FPathFollowingResultFlags::None));
			}
		}
		else if (bCanUpdatePathPartInTick)
		{
			// override radius multiplier and switch to next path part when closer than 4x agent radius
			const float NextPartMultiplier = 4.0f;
			const bool bHasReached = HasReachedInternal(GoalLocation, 0.0f, 0.0f, CurrentLocation, 0.0f, NextPartMultiplier);

			if (bHasReached)
			{
				SwitchToNextPathPart();
			}
		}
	}

	if (bCanReachTarget && Status == EPathFollowingStatus::Moving)
	{
		// check waypoint switch condition in meta paths
		FMetaNavMeshPath* MetaNavPath = bIsUsingMetaPath ? Path->CastPath<FMetaNavMeshPath>() : nullptr;
		if (MetaNavPath && Status == EPathFollowingStatus::Moving)
		{
			MetaNavPath->ConditionalMoveToNextSection(CurrentLocation, EMetaPathUpdateReason::MoveTick);
		}

		// gather location samples to detect if moving agent is blocked
		const bool bHasNewSample = UpdateBlockDetection();
		if (bHasNewSample && IsBlocked())
		{
			OnPathFinished(FPathFollowingResult(EPathFollowingResult::Blocked, FPathFollowingResultFlags::None));
		}
	}
}

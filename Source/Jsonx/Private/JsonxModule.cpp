// Copyright Epic Games, Inc. All Rights Reserved.

#include "CoreMinimal.h"
#include "JsonxGlobals.h"
#include "Modules/ModuleInterface.h"
#include "Modules/ModuleManager.h"


DEFINE_LOG_CATEGORY(LogJsonx);


/**
 * Implements the Jsonx module.
 */
class FJsonxModule
	: public IModuleInterface
{
public:

	// IModuleInterface interface

	virtual void StartupModule( ) override { }
	virtual void ShutdownModule( ) override { }

	virtual bool SupportsDynamicReloading( ) override
	{
		return false;
	}
};


IMPLEMENT_MODULE(FJsonxModule, Jsonx);

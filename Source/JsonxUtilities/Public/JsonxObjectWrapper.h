// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "UObject/ObjectMacros.h"
#include "UObject/Object.h"
#include "UObject/Class.h"

#include "JsonxObjectWrapper.generated.h"

class FJsonxObject;

/** UStruct that holds a JsonxObject, can be used by structs passed to JsonxObjectConverter to pass through JsonxObjects directly */
USTRUCT(BlueprintType)
struct JSONXUTILITIES_API FJsonxObjectWrapper
{
	GENERATED_USTRUCT_BODY()
public:

	UPROPERTY(EditAnywhere, Category = "JSONX")
	FString JsonxString;

	TSharedPtr<FJsonxObject> JsonxObject;

	bool ImportTextItem(const TCHAR*& Buffer, int32 PortFlags, UObject* Parent, FOutputDevice* ErrorText);
	bool ExportTextItem(FString& ValueStr, FJsonxObjectWrapper const& DefaultValue, UObject* Parent, int32 PortFlags, UObject* ExportRootScope) const;
	void PostSerialize(const FArchive& Ar);

	explicit operator bool() const
	{
		return JsonxObject.IsValid();
	}

	bool JsonxObjectToString(FString& Str) const;
	bool JsonxObjectFromString(const FString& Str);
};

template<>
struct TStructOpsTypeTraits<FJsonxObjectWrapper> : public TStructOpsTypeTraitsBase2<FJsonxObjectWrapper>
{
	enum
	{
		WithImportTextItem = true,
		WithExportTextItem = true,
		WithPostSerialize = true,
	};
};
UCLASS()
class UJsonxUtilitiesDummyObject : public UObject
{
	GENERATED_BODY()
};
